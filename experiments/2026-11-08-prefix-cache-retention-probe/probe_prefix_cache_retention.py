#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stress probe for prefix-cache retention around block boundaries and under
KV block eviction pressure across 64K-100K multi-turn agent workloads (#902).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from dataclasses import replace
from pathlib import Path

import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_group_id,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

HASH = 128
PAGE = 256
WINDOW = 128


def make_ced_manager(
    retention: int | None,
    *,
    num_blocks: int = 4096,
    max_model_len: int = 131072,
):
    init_none_hash(sha256)
    global_spec = MLAAttentionSpec(
        block_size=PAGE,
        num_kv_heads=1,
        head_size=512,
        tokens_per_state=2,
        dtype=torch.uint8,
    )
    encoder_spec = SlidingWindowMLASpec(
        block_size=HASH,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=WINDOW,
    )
    decoder_spec = replace(
        encoder_spec, prefix_cache_enabled=False, prefill_replay_window=WINDOW
    )
    groups = [
        KVCacheGroupSpec(["global-mla"], global_spec),
        KVCacheGroupSpec(["indexer-mla"], replace(global_spec, head_size=68)),
        KVCacheGroupSpec(["encoder-swa"], encoder_spec),
        KVCacheGroupSpec(["decoder-swa"], decoder_spec),
        KVCacheGroupSpec(
            ["draft-swa"], replace(decoder_spec, extra_retained_tokens=1)
        ),
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=retention,
    )
    manager = make_kv_cache_manager(
        config, max_model_len=max_model_len, hash_block_size=HASH
    )
    roles = {group.layer_names[0]: i for i, group in enumerate(groups)}
    return manager, roles


def make_ced_request(name: str, tokens: list[int], max_tokens: int = 8192):
    req = make_request(name, list(tokens), HASH, sha256)
    req.max_tokens = max_tokens
    req.sampling_params.max_tokens = max_tokens
    return req


def run_prefill_and_decode(
    manager,
    req,
    decode_tokens: list[int],
    *,
    tokens_per_step: int = 1,
    prefill_chunk_size: int | None = None,
) -> int:
    computed_blocks, num_cached_tokens, _ = manager.get_computed_blocks(req)
    remaining_prefill = req.num_prompt_tokens - num_cached_tokens
    if prefill_chunk_size is None or remaining_prefill <= prefill_chunk_size:
        if remaining_prefill > 0:
            assert (
                manager.allocate_slots(
                    req, remaining_prefill, num_cached_tokens, computed_blocks
                )
                is not None
            )
            req.num_computed_tokens = req.num_prompt_tokens
            manager.cache_blocks(req, req.num_computed_tokens)
        else:
            assert (
                manager.allocate_slots(req, 0, num_cached_tokens, computed_blocks)
                is not None
            )
            req.num_computed_tokens = num_cached_tokens
    else:
        first_chunk = min(prefill_chunk_size, remaining_prefill)
        assert (
            manager.allocate_slots(
                req, first_chunk, num_cached_tokens, computed_blocks
            )
            is not None
        )
        req.num_computed_tokens = num_cached_tokens + first_chunk
        manager.cache_blocks(req, req.num_computed_tokens)
        while req.num_computed_tokens < req.num_prompt_tokens:
            step = min(
                prefill_chunk_size, req.num_prompt_tokens - req.num_computed_tokens
            )
            assert manager.allocate_slots(req, step) is not None
            req.num_computed_tokens += step
            manager.cache_blocks(req, req.num_computed_tokens)

    idx = 0
    while idx < len(decode_tokens):
        step_tokens = decode_tokens[idx : idx + tokens_per_step]
        idx += len(step_tokens)
        req.append_output_token_ids(step_tokens)
        new_to_allocate = req.num_tokens - req.num_computed_tokens
        assert manager.allocate_slots(req, new_to_allocate) is not None
        req.num_computed_tokens = req.num_tokens
        manager.cache_blocks(req, req.num_computed_tokens)

    manager.free(req)
    return num_cached_tokens


def run_boundary_matrix_probe() -> dict:
    t0 = time.perf_counter()
    retentions = [0, 512, 8192, None]
    prompt_bases = [256, 512, 1024]
    prompt_offsets = [-1, 0, 1]
    decode_stop_offsets = [127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 641]
    step_sizes = [1, 3, 6]

    total_cases = 0
    passed_ext_cases = 0
    passed_retry_cases = 0
    swa_slid_past_boundary_cases = 0
    swa_slid_past_boundary_passed = 0

    for retention in retentions:
        for base in prompt_bases:
            for poff in prompt_offsets:
                for doff in decode_stop_offsets:
                    for step in step_sizes:
                        manager, _ = make_ced_manager(retention, num_blocks=2048)
                        prompt_len = base + poff
                        target_total = base + doff
                        all_tokens = list(range(target_total + 512))
                        req1 = make_ced_request("t1", all_tokens[:prompt_len])
                        run_prefill_and_decode(
                            manager,
                            req1,
                            all_tokens[prompt_len:target_total],
                            tokens_per_step=step,
                        )
                        req2 = make_ced_request("t2", all_tokens[: target_total + 256])
                        _, hit_ext, _ = manager.get_computed_blocks(req2)
                        expected_ext = (target_total // PAGE) * PAGE

                        req_retry = make_ced_request("retry", all_tokens[:target_total])
                        _, hit_retry, _ = manager.get_computed_blocks(req_retry)
                        expected_retry = ((target_total - WINDOW) // PAGE) * PAGE

                        total_cases += 1
                        if hit_ext == expected_ext:
                            passed_ext_cases += 1
                        if hit_retry == expected_retry:
                            passed_retry_cases += 1
                        if (target_total % PAGE) >= WINDOW and retention is not None:
                            swa_slid_past_boundary_cases += 1
                            if hit_ext == expected_ext:
                                swa_slid_past_boundary_passed += 1

    # Also measure intermediate SWA recycling during a 4096-token decode
    recycle_summary = {}
    for retention in [0, 512, 8192, None]:
        manager, roles = make_ced_manager(retention, num_blocks=2048)
        swa_gid = roles["encoder-swa"]
        prompt_len = 513
        total_len = prompt_len + 4096
        tokens = list(range(total_len + 256))
        req = make_ced_request("long", tokens[:prompt_len])
        run_prefill_and_decode(manager, req, tokens[prompt_len:total_len], tokens_per_step=4)
        cached_swa = sum(
            1
            for b in manager.block_pool.blocks
            if b.block_hash is not None and get_group_id(b.block_hash) == swa_gid
        )
        req2 = make_ced_request("long-t2", tokens[: total_len + 256])
        _, hit, _ = manager.get_computed_blocks(req2)
        recycle_summary[str(retention)] = {
            "cached_swa_blocks": cached_swa,
            "turn2_hit_tokens": hit,
            "expected_hit_tokens": (total_len // PAGE) * PAGE,
        }

    elapsed_s = time.perf_counter() - t0
    return {
        "total_cases": total_cases,
        "passed_ext_cases": passed_ext_cases,
        "passed_retry_cases": passed_retry_cases,
        "swa_slid_past_boundary_cases": swa_slid_past_boundary_cases,
        "swa_slid_past_boundary_passed": swa_slid_past_boundary_passed,
        "long_decode_swa_recycle": recycle_summary,
        "elapsed_s": round(elapsed_s, 4),
    }


def run_spec_decode_scheduler_probe() -> dict:
    t0 = time.perf_counter()
    block_size = 16
    total_cases = 0
    passed_cases = 0
    crossing_cases = 0
    crossing_passed = 0

    for async_scheduling in [False, True]:
        for num_spec in [1, 2, 3, 4, 5]:
            for offset in [1, 2, 3, 4, 5, 6]:
                scheduler = create_scheduler(
                    enable_prefix_caching=True,
                    block_size=block_size,
                    num_speculative_tokens=num_spec,
                    speculative_method="ngram_gpu" if async_scheduling else "ngram",
                    async_scheduling=async_scheduling,
                )
                prompt_len = 2 * block_size - offset
                total_gen_step2 = min(1 + num_spec, max(1, offset))
                max_tokens = 1 + total_gen_step2
                will_cross = (prompt_len + max_tokens - 1) >= 2 * block_size

                req1 = create_requests(
                    num_requests=1,
                    num_tokens=prompt_len,
                    max_tokens=max_tokens,
                    block_size=block_size,
                    same_prompt=True,
                )[0]
                scheduler.add_request(req1)

                sched_out1 = scheduler.schedule()
                drafts = [0] * num_spec
                model_out1 = ModelRunnerOutput(
                    req_ids=[req1.request_id],
                    req_id_to_index={req1.request_id: 0},
                    sampled_token_ids=[[0]],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                )
                scheduler.update_from_output(sched_out1, model_out1)

                if not req1.is_finished():
                    scheduler.update_draft_token_ids(
                        DraftTokenIds([req1.request_id], [list(drafts)])
                    )
                    sched_out2 = scheduler.schedule()
                    model_out2 = ModelRunnerOutput(
                        req_ids=[req1.request_id],
                        req_id_to_index={req1.request_id: 0},
                        sampled_token_ids=[[0] * total_gen_step2],
                        logprobs=None,
                        prompt_logprobs_dict={},
                        pooler_output=[],
                    )
                    scheduler.update_from_output(sched_out2, model_out2)

                req2 = create_requests(
                    num_requests=1,
                    num_tokens=2 * block_size + 4,
                    max_tokens=4,
                    block_size=block_size,
                    req_ids=["followup"],
                    same_prompt=True,
                )[0]
                _, num_cached, _ = scheduler.kv_cache_manager.get_computed_blocks(req2)
                expected = 2 * block_size if will_cross else block_size
                total_cases += 1
                if num_cached == expected:
                    passed_cases += 1
                if will_cross:
                    crossing_cases += 1
                    if num_cached == expected:
                        crossing_passed += 1

    elapsed_s = time.perf_counter() - t0
    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "crossing_cases": crossing_cases,
        "crossing_passed": crossing_passed,
        "elapsed_s": round(elapsed_s, 4),
    }


def run_lru_ordering_probe() -> dict:
    t0 = time.perf_counter()
    results = {}
    for retention in [0, 512, 8192, None]:
        manager, roles = make_ced_manager(retention, num_blocks=1024)
        swa_gid = roles["encoder-swa"]
        tokens = list(range(32768))

        req1 = make_ced_request("turn-1", tokens[:16384])
        run_prefill_and_decode(manager, req1, tokens[16384:16896], tokens_per_step=4)

        req2 = make_ced_request("turn-2", tokens[:24576])
        hit2 = run_prefill_and_decode(manager, req2, tokens[24576:25088], tokens_per_step=4)

        hash_to_end = {
            blk_hash: (idx_blk + 1) * HASH
            for idx_blk, blk_hash in enumerate(req2.block_hashes)
        }
        free_blocks = manager.block_pool.free_block_queue.get_all_free_blocks()
        cached_entries = []
        for blk in free_blocks:
            if blk.block_hash is None:
                continue
            gid = get_group_id(blk.block_hash)
            raw_hash = get_block_hash(blk.block_hash)
            end_tok = hash_to_end[raw_hash]
            is_sparse = gid == swa_gid
            cached_entries.append((end_tok, is_sparse, gid, blk.block_id))

        inversions = 0
        for i in range(len(cached_entries) - 1):
            end_a, sparse_a, _, _ = cached_entries[i]
            end_b, sparse_b, _, _ = cached_entries[i + 1]
            if end_a < end_b or (end_a == end_b and sparse_a > sparse_b):
                inversions += 1

        results[str(retention)] = {
            "turn2_hit_tokens": hit2,
            "num_cached_blocks_in_queue": len(cached_entries),
            "first_evict_end_token": cached_entries[0][0] if cached_entries else None,
            "last_evict_end_token": cached_entries[-1][0] if cached_entries else None,
            "lru_inversions": inversions,
        }

    elapsed_s = time.perf_counter() - t0
    return {
        "by_retention": results,
        "elapsed_s": round(elapsed_s, 4),
    }


def run_64k_to_100k_agent_stress_probe() -> dict:
    t0 = time.perf_counter()
    profiles = {
        "tp3-triangle": 512,
        "tp4-ring": 8192,
    }
    turn_schedule = [
        (65536, 389, 4),   # Turn 1: 65,536 -> 65,925 (65925 % 256 == 133 >= 128)
        (70144, 412, 6),   # Turn 2: 70,144 -> 70,556 (70556 % 256 == 156 >= 128)
        (75008, 290, 3),   # Turn 3: 75,008 -> 75,298 (75298 % 256 == 34 < 128)
        (80128, 395, 5),   # Turn 4: 80,128 -> 80,523 (80523 % 256 == 139 >= 128)
        (85248, 401, 4),   # Turn 5: 85,248 -> 85,649 (85649 % 256 == 145 >= 128)
        (90112, 310, 6),   # Turn 6: 90,112 -> 90,422 (90422 % 256 == 54 < 128)
        (95232, 399, 5),   # Turn 7: 95,232 -> 95,631 (95631 % 256 == 143 >= 128)
        (99584, 512, 4),   # Turn 8: 99,584 -> 100,096 (100,096 tokens)
    ]

    profile_results = {}
    for profile_name, retention in profiles.items():
        p_t0 = time.perf_counter()
        manager, _ = make_ced_manager(
            retention,
            num_blocks=2200,
            max_model_len=131072,
        )
        num_agents = 2
        agent_tokens = {
            agent_id: [agent_id * 1_000_000 + i for i in range(102400)]
            for agent_id in range(num_agents)
        }
        prev_completed_len = {agent_id: 0 for agent_id in range(num_agents)}

        total_reusable = 0
        total_hit = 0
        no_evict_rates = []
        evict_rates = []
        turn_records = []

        for turn_idx, (prompt_len, decode_len, step_size) in enumerate(turn_schedule):
            eviction_active = turn_idx in (3, 5, 7)
            if eviction_active:
                free_list = manager.block_pool.free_block_queue.get_all_free_blocks()
                blocks_to_claim = 0
                cached_seen = 0
                for b in free_list:
                    blocks_to_claim += 1
                    if b.block_hash is not None:
                        cached_seen += 1
                        if cached_seen == 18:
                            break
                claimed = manager.block_pool.get_new_blocks(blocks_to_claim)
                for b in claimed:
                    b.ref_cnt -= 1
                manager.block_pool.free_block_queue.prepend_n(claimed)

            for agent_id in range(num_agents):
                seq = agent_tokens[agent_id]
                prompt = seq[:prompt_len]
                decode = seq[prompt_len : prompt_len + decode_len]
                req = make_ced_request(f"agent-{agent_id}-turn-{turn_idx}", prompt)
                hit = run_prefill_and_decode(
                    manager,
                    req,
                    decode,
                    tokens_per_step=step_size,
                    prefill_chunk_size=8192,
                )
                prev_len = prev_completed_len[agent_id]
                if prev_len > 0:
                    max_possible = (prev_len // PAGE) * PAGE
                    ratio = hit / prev_len
                    total_reusable += prev_len
                    total_hit += hit
                    is_evicted_agent = eviction_active and agent_id == 0
                    if is_evicted_agent:
                        evict_rates.append(ratio)
                    else:
                        no_evict_rates.append(ratio)
                    turn_records.append(
                        {
                            "turn": turn_idx + 1,
                            "agent_id": agent_id,
                            "prompt_len": prompt_len,
                            "completed_len": prompt_len + decode_len,
                            "prev_completed_len": prev_len,
                            "max_aligned_prefix": max_possible,
                            "hit_tokens": hit,
                            "hit_rate": round(ratio, 6),
                            "eviction_applied_before_turn": is_evicted_agent,
                        }
                    )
                prev_completed_len[agent_id] = prompt_len + decode_len

        overall_rate = total_hit / total_reusable
        profile_results[profile_name] = {
            "retention_interval": retention,
            "total_reusable_tokens": total_reusable,
            "total_hit_tokens": total_hit,
            "overall_hit_rate": round(overall_rate, 6),
            "mean_non_evicted_hit_rate": round(
                sum(no_evict_rates) / len(no_evict_rates), 6
            ),
            "min_non_evicted_hit_rate": round(min(no_evict_rates), 6),
            "mean_evicted_turn_hit_rate": round(
                sum(evict_rates) / len(evict_rates), 6
            ),
            "min_evicted_turn_hit_rate": round(min(evict_rates), 6),
            "turns": turn_records,
            "elapsed_s": round(time.perf_counter() - p_t0, 4),
        }

    return {
        "profiles": profile_results,
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only"
    )
    payload = {
        "boot": args.boot,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "boundary_matrix_probe": run_boundary_matrix_probe(),
        "spec_decode_scheduler_probe": run_spec_decode_scheduler_probe(),
        "lru_ordering_probe": run_lru_ordering_probe(),
        "agent_64k_to_100k_stress_probe": run_64k_to_100k_agent_stress_probe(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + chr(10), encoding="utf-8")
    print(f"Wrote probe receipt to {args.output}")
    for prof, data in payload["agent_64k_to_100k_stress_probe"]["profiles"].items():
        print(
            f"  {prof} (retention={data['retention_interval']}): "
            f"overall={data['overall_hit_rate']:.4%}, "
            f"non_evicted_mean={data['mean_non_evicted_hit_rate']:.4%}, "
            f"evicted_mean={data['mean_evicted_turn_hit_rate']:.4%}, "
            f"evicted_min={data['min_evicted_turn_hit_rate']:.4%}"
        )


if __name__ == "__main__":
    main()
