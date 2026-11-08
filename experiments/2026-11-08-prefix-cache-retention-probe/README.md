# Prefix-Cache Block-Boundary & Eviction Retention (`#902`)

- **Date:** 2026-11-08
- **Issue:** [`#902`](https://github.com/christopherowen/spark-ds41f/issues/902) (`TODO.md` lines 275–278)
- **Base commit:** `8e5ea208` (`2026-11-07-spark-ds41f-1107`, vLLM 641-patch head `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate patch:** [`vllm/0762-fix-prefix-cache-retain-decode-frontier-swa-and-orde.patch`](vllm/0762-fix-prefix-cache-retain-decode-frontier-swa-and-orde.patch) (`patch_head: 34757a313c2b317877cb7111744e2d6363d5145a`, `expected_tree: 03cb8b23c4f238f27cda0232ed0dca2cc610dd9e`)
- **Verification node:** `sp7` (`spark-0015`, `NVIDIA GB10`, container `spark-ds41f:1107`, 2 fresh container boots serialized under `flock /tmp/gpu.lock`)

---

## 1. Root-Cause Analysis

Multi-turn `64K–100K` agent workloads on DeepSeek-V4.1-Flash combine three hybrid KV cache groups sharing one `BlockPool` (`alignment_tokens = lcm(256, 128) = 256`):
- **Group 0 (`FullAttentionManager`):** MLA compressed KV (`block_size = 256`), dense retention.
- **Group 1 (`SlidingWindowManager`):** SWA local window (`block_size = 128`, `sliding_window = 128`, `sliding_window_contiguous_blocks = 1`), sparse checkpoint retention (`prefix_cache_retention_interval = 512` on `tp3-triangle`, `8192` on `tp4-ring`).
- **Group 2 (`FullAttentionManager`):** KDA / full-attention layers (`block_size = 256`).

Under `HybridKVCacheCoordinator.find_longest_cache_hit`, a prefix hit of length $L$ (a multiple of `256`) requires:
1. FullAttention Group 0 & Group 2 blocks cached across `[0, L)`.
2. SlidingWindow Group 1 block at `[L - 128, L)` cached (so the 128-token local sliding window at the boundary of $L$ is valid without recomputing from $0$).

Three bugs in `vllm/v1/core/` caused prefix-cache hit rates on `64K–100K` multi-turn agent streams to collapse below the `98–99%` target:

### Bug 1: Decode-Frontier SWA Checkpoint Loss (`SlidingWindowManager.cache_blocks`)
`SparseRetentionPolicy._retain(end_tokens, request)` only returned `True` at sparse segment multiples (`end_tokens % interval == 0`) and at the prompt endpoint (`end_tokens == request.num_prompt_tokens`). During decode (`num_tokens > request.num_prompt_tokens`), as a request crossed a 256-token boundary $B = 256 k$ (which is not a multiple of `8192` or `512`) and then decoded past $B + 128$ (`T_end % 256 >= 128`):
- `SlidingWindowManager.cache_blocks` skipped caching the SWA block ending at $B$ (`[B - 128, B)`).
- Once decode reached $B + 128$, `[B - 128, B)` left the 128-token sliding window and `SlidingWindowManager.remove_skipped_blocks` freed it (`reset_hash()` + return to `free_block_queue`) before `free()` ever ran.
- When Turn $k$ finished at $T_{\text{end}}$ with $T_{\text{end}} \bmod 256 \ge 128$, `retain_endpoint(request, T_end)` only saw the still-live SWA block `[B, B + 128)` (whose end token $B + 128$ is an odd multiple of `128`, not aligned to the 256-token MLA boundary $B$).
- On Turn $k+1$, `HybridKVCacheCoordinator.find_longest_cache_hit` could not match $B$ because the SWA block at `[B - 128, B)` had already been freed un-hashed, losing up to an entire `8,192`-token (`tp4-ring`) or `512`-token (`tp3-triangle`) segment of decoded/prefilled history.

### Bug 2: Inverted Root-Before-Tail LRU Eviction Order in the Shared `BlockPool`
When concurrent `64K–100K` agent streams exerted eviction pressure on the shared `BlockPool`, the oldest shared prefix checkpoints (`512`, `8,192`, `16,384`) were evicted **before** the newest tail blocks (`96K–100K`):
1. `SlidingWindowManager.remove_skipped_blocks` freed retained SWA checkpoints in **ascending** token order as the sliding window advanced (`512`, then `1024`, ..., up to `96K`), placing early SWA checkpoints at the **LRU head** of `free_block_queue` (first to be evicted).
2. `SlidingWindowManager.add_local_computed_blocks` only touched the 128-token window tail `[hit - 128, hit)` on a prefix-cache hit, leaving all earlier retained SWA checkpoints (`512, 1024, ... < hit - 128`) of that matched prefix un-refreshed at the LRU head of `free_block_queue`.
3. `HybridKVCacheCoordinator.free` freed all Group 0 (`FullAttentionManager`) blocks `[T..0]` first, followed by Group 1 (`SlidingWindowManager`) blocks, rather than interleaving across groups by descending token position. Worse, evicting a single retained SWA checkpoint at `8,192` invalidated `8,192` tokens of prefix reuse (`64` blocks), whereas evicting a 256-token MLA block only invalidated `256` tokens.

### Bug 3: Un-Cached Block Boundary Crossing on Speculative Decode Stopping Steps
In `Scheduler._update_request_with_output`, when a running request crossed a 256-token block boundary via accepted speculative draft tokens (`num_computed_tokens > num_tokens_before`) and finished on that same step (`stopped = True`), `cache_blocks` was only called in the `else` (non-stopped) branch inside `schedule()`, never before `_free_request(request)`. Any full block completed on the final speculative decode step was freed un-hashed. In `AsyncScheduler._update_request_with_output`, `cache_blocks` also lacked a `not is_stale` guard and `min(request.num_computed_tokens, request.num_tokens)` clamp.

---

## 2. Changes (`vllm` Patch `0762`)

1. **Rolling Decode-Frontier SWA Retention & Intermediate Frontier Recycling (`vllm/v1/core/single_type_kv_cache_manager.py`):**
   - In `SlidingWindowManager.cache_blocks`, compute the rolling decode-frontier boundary `frontier_boundary = (num_computed_tokens // 256) * 256` (`resend_lo = frontier_boundary - 128`, `resend_hi = frontier_boundary`, `aligned_ext = (num_computed_tokens // 128) * 128`) whenever `num_computed_tokens > request.num_prompt_tokens`.
   - Retain SWA blocks at `(resend_lo, resend_hi, aligned_ext)` across 256-token decode crossings so `[B - 128, B)` remains hashed when `remove_skipped_blocks` advances past $B + 128$.
   - Track intermediate non-segment decode-frontier SWA blocks per request in `_req_decode_frontier_blocks[request_id]`; as soon as decode advances to a newer 256-token boundary $B' > B$, recycle obsolete intermediate SWA blocks from $B$ (`ref_cnt == 0`) by removing them from `cached_block_hash_to_block`, resetting their hash, and prepending them to the scratch front of `free_block_queue` via `prepend_n`. Only the final turn endpoint SWA checkpoint remains cached when the turn completes.
2. **Tail-to-Root Hybrid LRU Eviction Ordering (`vllm/v1/core/single_type_kv_cache_manager.py`, `vllm/v1/core/kv_cache_coordinator.py`):**
   - Track retained SWA blocks per request (`_req_retained_blocks`) and prefix-chain checkpoint inheritance (`_block_prefix_chain`) in `SlidingWindowManager`.
   - On `add_local_computed_blocks` (cache hit), `remove_skipped_blocks` (sliding window advance), and `free`, refresh free (`ref_cnt == 0`) retained SWA checkpoints in **descending** token order (`_refresh_free_retained_blocks_desc`) so `81,920` sits toward the LRU head (evicted first) and root checkpoints (`512`, `8,192`) sit at the MRU tail (evicted last).
   - In `KVCacheCoordinator.pop_blocks_for_free` and `free`, interleave blocks across all KV cache groups sorted by `(-end_token, is_sparse, group_id, -blk_idx)`: highest token positions are freed first (LRU head, evicted first), lowest token positions (`0..256`) are freed last (MRU tail, evicted last), and at equal `end_token` dense full-attention blocks are freed before sparse SWA segment checkpoints.
3. **Speculative Decode Stopping-Step Block Caching (`vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/sched/async_scheduler.py`):**
   - In `Scheduler._update_request_with_output` and `AsyncScheduler._update_request_with_output`, call `self.kv_cache_manager.cache_blocks(request, min(request.num_computed_tokens, request.num_tokens))` whenever `status_before_update == RequestStatus.RUNNING and not is_stale and request.num_output_placeholders == 0 and request.num_computed_tokens > num_tokens_before`.

---

## 3. 2-Boot Verification on `sp7` (`NVIDIA GB10`)

All runs were executed inside fresh `spark-ds41f:1107` containers on `sp7` (`spark-0015`, `NVIDIA GB10`) serialized with `flock /tmp/gpu.lock`.

### 3.1 Multi-Turn `64K–100K` Concurrent Agent Stress Probe (`4` agents × `8` turns, `64K -> 100K` tokens, `2,444,316` reusable prefix tokens)

| Profile | `prefix_cache_retention_interval` | Metric | Before Fix (`1107` Base) | Boot 1 (`0762`) | Boot 2 (`0762`) | Mean ± 95% CI |
|---|---:|---|---:|---:|---:|---:|
| `tp3-triangle` | `512` | Overall reusable prefix hit rate | `57.09%` | **`99.3809%`** (`2,429,184 / 2,444,316`) | **`99.3809%`** (`2,429,184 / 2,444,316`) | **`99.3809% ± 0.0000%`** |
| `tp3-triangle` | `512` | Non-evicted turn hit rate (mean / min) | `72.41%` / `0.56%` | **`99.8478%`** / **`99.6223%`** | **`99.8478%`** / **`99.6223%`** | **`99.8478% ± 0.0000%`** |
| `tp3-triangle` | `512` | Evicted turn hit rate (mean / min) | `0.00%` / `0.00%` | **`97.7496%`** / **`97.2350%`** | **`97.7496%`** / **`97.2350%`** | **`97.7496% ± 0.0000%`** |
| `tp4-ring` | `8192` | Overall reusable prefix hit rate | `56.34%` | **`98.9723%`** (`2,419,200 / 2,444,316`) | **`98.9723%`** (`2,419,200 / 2,444,316`) | **`98.9723% ± 0.0000%`** |
| `tp4-ring` | `8192` | Non-evicted turn hit rate (mean / min) | `71.46%` / `0.00%` | **`99.8478%`** / **`99.6223%`** | **`99.8478%`** / **`99.6223%`** | **`99.8478% ± 0.0000%`** |
| `tp4-ring` | `8192` | Evicted turn hit rate (mean / min) | `0.00%` / `0.00%` | **`95.9408%`** / **`93.4952%`** | **`95.9408%`** / **`93.4952%`** | **`95.9408% ± 0.0000%`** |

### 3.2 Block-Boundary, Speculative Decode, Intermediate SWA Recycling, and LRU Ordering Suite

| Probe / Test Suite | Before Fix (`1107` Base) | Boot 1 (`0762`) | Boot 2 (`0762`) |
|---|---:|---:|---:|
| Exact Block-Boundary Matrix (`block_size - 1`, `block_size`, `block_size + 1`, `383..385`, `511..513`, `8191..8193`, spec-decode steps `1, 2, 3, 5, 6`, `retention in {0, 512, 8192, None}`) | `1,134 / 1,404` (`270` failures when `T_end % 256 >= 128`) | **`1,404 / 1,404` (`100%`)** | **`1,404 / 1,404` (`100%`)** |
| Long-Decode (`4,096` tokens) Cached SWA Blocks (`retention=0 / 512 / 8192 / None`) | Missing endpoint when `T_end % 256 >= 128` | **`4 / 11 / 4 / 36` blocks (`100%` Turn 2 hit)** | **`4 / 11 / 4 / 36` blocks (`100%` Turn 2 hit)** |
| Speculative Decode (`1..6` tok/step) Boundary Crossing on Stopping Step (`Scheduler` + `AsyncScheduler`) | Failed on `Scheduler` stopping step | **`60 / 60` (`100%`)** | **`60 / 60` (`100%`)** |
| Shared `BlockPool` Tail-to-Root LRU Eviction Order (`FullAttention` + `SlidingWindow`) | Root SWA evicted before tail | **`0` inversions** (`first=25,088`, `last=256`) | **`0` inversions** (`first=25,088`, `last=256`) |
| Full GPU `pytest` (`test_prefix_cache_retention_probe.py`, `test_prefix_caching.py`, `test_kv_cache_utils.py`, `test_FullAttentionManager.py`, `test_scheduler_Scenarios.py`, `test_async_scheduler.py`) | — | **`2,679 passed in 110.14s`** | **`2,679 passed in 110.20s`** |

---

## 4. Exact Reproduction Commands

```bash
# 1. Run standalone stress probe inside container on sp7
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -v /home/chris/projects/spark-ds41f:/workspace \
  spark-ds41f:1107 \
  python3 /workspace/experiments/2026-11-08-prefix-cache-retention-probe/probe_prefix_cache_retention.py \
  --json-out /workspace/experiments/2026-11-08-prefix-cache-retention-probe/receipts/probe-boot1.json

# 2. Run full v1 core KV cache + scheduler pytest suite inside container on sp7
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -v /home/chris/projects/spark-ds41f:/workspace \
  spark-ds41f:1107 \
  pytest -v \
    tests/v1/core/test_prefix_cache_retention_probe.py \
    tests/v1/core/test_prefix_caching.py \
    tests/v1/core/test_kv_cache_utils.py \
    tests/v1/core/test_FullAttentionManager.py \
    tests/v1/core/test_scheduler_Scenarios.py \
    tests/v1/core/test_async_scheduler.py
```

## 5. Receipts

- [`receipts/probe-boot1.json`](receipts/probe-boot1.json)
- [`receipts/probe-boot2.json`](receipts/probe-boot2.json)
- [`receipts/gpu-pytest-boot1.txt`](receipts/gpu-pytest-boot1.txt)
- [`receipts/gpu-pytest-boot2.txt`](receipts/gpu-pytest-boot2.txt)
