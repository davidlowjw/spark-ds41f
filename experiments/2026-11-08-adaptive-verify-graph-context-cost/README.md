# Experiment: Context-Length Band, 16-Row MMA Tile, and 64-Row Wave-Saturation Pricing in `AdaptiveVerificationManager._step_cost`

## Metadata

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **Base vLLM patch head**: `320021e3c4dbb202d911ab70e2cd924c75420488` (`0761`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM patch**: `experiments/2026-11-08-adaptive-verify-graph-context-cost/vllm/0762-perf-spec-decode-adaptive-verify-graph-context-cost.patch`
- **Candidate vLLM patch head**: `970944cee9c2276e6533e1d10eb1ac47a3d7b07f` (tree `1fc232524f96810cb121c3941dc255f5a0a43567`, `642` active patches)
- **Hardware**: `sp5` and `sp8` (`NVIDIA GB10`, 48 SMs, `sm_121a`, container `spark-ds41f:1107`)
- **Causal variable**: Replace the unbounded two-segment (`<= 16` at `1.0x`, `> 16` at `0.55x`) context-blind step-cost model in `AdaptiveVerificationManager._step_cost` (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`) with a deterministic, rank-invariant three-regime MMA-tile / 64-row wave-saturation + context-length band cost model (`_step_cost`, `_context_row_cost_scale`, `effective_dead_rows_tau`, `get_num_tokens`).

## Motivation & Hardware Diagnosis (`NVIDIA GB10`)

In `spark-ds41f-1107` (following patches `0715`, `0729`, and `0742`), `AdaptiveVerificationManager._step_cost` priced verification rows `R` as:
$$\text{cost}(R) = 1.0 + \text{row\_cost} \cdot \left(\min(R, 16) + 0.55 \cdot \max(R - 16, 0)\right)$$
This left three physical regimes on `NVIDIA GB10` (48 SMs, `sm_121a`) unmodeled:

1. **16-Row MMA Tile & Decode Kernel Transitions (`R = 16 -> 17` and `R = 32 -> 33`)**:
   - In `vllm/models/deepseek_v4_1/tilelang/linear.py` and `gemm.py`, dense MXFP8 projections dispatch discrete row-tile kernels (`decode_16` for `R <= 16`, `decode_32` for `17 <= R <= 32`, `decode_64` for `33 <= R <= 64`, and `prefill_128` for `R > 64`).
   - Across 40 layers of Dense FP8 projections (`fused_wqa_wkv`, `wq_b`, `wo_b`, `shared_gate_up`, `shared_down`) measured across 2 boots on GB10 (`receipts/attn-idx-profile-boot1.json`, `receipts/attn-idx-profile-boot2.json`):
     - `R = 1..16` (`decode_16`, 1 MMA tile): `2,182 -> 1,482 us`
     - `R = 16 -> 17` (`decode_16 -> decode_32`, 2 MMA tiles): **`1,482 -> 2,681 us` (`+1,199 us` / `+80.9%` step jump)**
     - `R = 32 -> 40` (`decode_32 -> decode_64`, 3–4 MMA tiles): **`1,933 -> 4,279 us` (`+2,346 us` / `+121.4%` step jump)**
2. **64-Row Second-Wave Saturation Knee (`R > 64`)**:
   - Applying `HIGH_ROW_COST_SCALE = 0.55` above `R = 16` with no upper bound caused `c16`/`c32` cohorts to drop the marginal draft survival threshold to `~0.33` at `R = 65..96`.
   - Above `R = 64` on 48 SMs, dense projections cross `DECODE_ROWS = 64` into `prefill_128` (`2,857 -> 5,425 us`, **`+2,569 us` / `+89.9%`** at `R = 80`), `IndexerV41Plan` crosses `DECODE_CHUNK = 64` into a second 64-row wave (`431 -> 726 us` at 2K, `4,070 -> 5,170 us` at 64K), and routed MoE (`64 * 6 = 384` expert slots across 256 experts) is already saturated.
3. **Uncoalesced Long-Context Indexer + Attention Scaling (`8K..128K`)**:
   - On single requests or moderate batches (`c1..c8`) at `32K..128K` context length, global KV page occupancy stays well below the 70% `KV_PRESSURE_KNEE`, so `kv_pressure == 0.0`.
   - Yet across the 8-layer `IndexerV41Plan` (`3` layers of `compress_ratio=2` full-index scan, `1` layer of `compress_ratio=1` candidate-source scan, and `4` layers of `compress_ratio=1` `CANDIDATE_BLOCKS = 256` reindex which saturates at `L = 16,384`), per-row indexer cost rises from `0.4..2.9 us/row` at `2K` to `23.7..30.0 us/row` at `16K`, `56.6..60.1 us/row` at `64K`, and **`96.9..103.5 us/row` at `128K`**. Crucially, this per-row context scan cost does **not** coalesce in the `17..64` row regime (`96.9 us/row` in `1..16` vs `103.5 us/row` in `17..64` at `128K`).

### Measured 2-Boot GB10 Kernel Latency Table (`receipts/attn-idx-profile-boot{1,2}.json`)

| Total Rows `R` | `40L_dense` (us, 2-boot mean) | `40L_mla` (us, 2-boot mean) | `8L_idx` @ 2K (us) | `8L_idx` @ 8K (us) | `8L_idx` @ 16K (us) | `8L_idx` @ 32K (us) | `8L_idx` @ 64K (us) | `8L_idx` @ 128K (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `2184.1` | `726.5` | `272.1` | `283.3` | `279.6` | `289.9` | `277.4` | `319.6` |
| `4` | `2054.2` | `753.0` | `278.0` | `279.8` | `282.5` | `304.3` | `401.7` | `631.8` |
| `8` | `2068.8` | `806.2` | `282.8` | `281.1` | `366.1` | `457.9` | `661.7` | `998.7` |
| `16` | `1482.4` | `1363.1` | `281.3` | `387.3` | `636.8` | `814.9` | `1126.2` | `1775.0` |
| `17` | **`2681.4`** | `1413.6` | `303.4` | `387.0` | `641.4` | `832.7` | `1171.5` | `1879.8` |
| `24` | `2261.8` | `2153.2` | `275.2` | `507.2` | `913.0` | `1132.4` | `1605.2` | `2563.2` |
| `32` | `1933.2` | `2504.5` | `291.3` | `646.6` | `1114.7` | `1428.6` | `2091.2` | `3425.3` |
| `40` | **`4279.5`** | `3049.6` | `314.3` | `779.0` | `1387.5` | `1758.5` | `2583.1` | `4231.7` |
| `48` | `3986.3` | `3671.6` | `349.4` | `938.3` | `1600.0` | `2090.1` | `3043.7` | `5033.8` |
| `64` | `2856.6` | `5076.1` | `430.9` | `1161.8` | `2084.7` | `2746.1` | `4070.2` | `6718.1` |
| `80` | **`5425.4`** | `6206.7` | **`725.9`** | `1593.4` | `2688.7` | `3527.7` | `5170.0` | `8501.1` |
| `96` | `4847.5` | `6943.3` | `729.9` | `1814.8` | `3178.2` | `4164.5` | `6184.2` | `10123.4` |

## Calibrated Step-Cost Model (`0762`)

Using only rank-invariant scheduler/worker metadata (`total_rows`, `num_reqs`, and `req_states.num_computed_tokens_np[slots]`):

1. **Three-Regime Row Cost + MMA Tile / Wave Steps (`_step_cost`)**:
   - `HIGH_ROW_COST_KNEE = 16`, `HIGH_ROW_COST_SCALE = 0.55` (`1.0` when `num_reqs == 1`).
   - `SATURATED_ROW_COST_KNEE = 64`, `SATURATED_ROW_COST_SCALE = 1.0` (restores full per-row slope above 64 rows where `DECODE_ROWS = 64`, `DECODE_CHUNK = 64`, and 48 SMs saturate).
   - Discrete tile/wave boundary steps:
     - Single-request (`num_reqs == 1`): `SINGLE_REQ_TILE_STEP_ROWS = 1.5` at `R > 16` (`decode_16 -> decode_32`).
     - Multi-request (`num_reqs > 1`): `MMA_TILE_STEP_ROWS = 0.25` at `R > 16` (`+0.25` rows) and `3 * MMA_TILE_STEP_ROWS` (`+0.75` rows) at `R > 32` (`decode_32 -> decode_64`).
     - Second wave (`R > 64`): `SECOND_WAVE_STEP_ROWS = 2.0` per 64-row wave above `SATURATED_ROW_COST_KNEE = 64`.
2. **Two-Slope Context-Length Row-Cost Scaling (`_context_row_cost_scale`)**:
   - `CONTEXT_ROW_COST_KNEE = 8192` (`1.0x` at `<= 8K`).
   - `CONTEXT_REINDEX_SATURATION = 16384`, `CONTEXT_REINDEX_WEIGHT = 0.12` (captures the 4-layer `compress_ratio=1` reindex ramp up to `CANDIDATE_BLOCKS * CANDIDATE_BLOCK = 16,384` tokens).
   - `CONTEXT_SOURCE_SPAN = 65536`, `CONTEXT_SOURCE_SLOPE = 0.22`, capped at `CONTEXT_MAX_ROW_COST_SCALE = 2.25` (captures the linear 3-layer `compress_ratio=2` + 1-layer `compress_ratio=1` source indexer scan).
   - Applied as uncoalesced per-row excess `context_excess_rows` in `_step_cost`, folded into `effective_dead_rows_tau(context_scale)`, and used via additive marginal surplus `valid_scores - row_cost * max(scale - 1, 0)` to prioritize short-context drafts in heterogeneous batches.

## 2-Boot Hardware Benchmark Results (`receipts/screen-graph-context-cost-boot{1,2}.jsonl`)

| Regime | Rows (`Base -> Cand`) | Boot 1 Step ms (`Base -> Cand`) | Boot 2 Step ms (`Base -> Cand`) | Boot 1 `tok/s` (`Base -> Cand`) | Boot 2 `tok/s` (`Base -> Cand`) | 2-Boot Mean Speedup (Range) |
|---|---:|---:|---:|---:|---:|---:|
| `c1_copy_burst_plateau_2k` | `20 -> 16` | `27.767 -> 25.467` | `27.673 -> 25.427` | `535.77 -> 542.28` (`+1.22%`) | `537.59 -> 543.13` (`+1.03%`) | **`+1.12%`** (`+1.03%..+1.22%`) |
| `c8_med_accept_2k` | `40 -> 40` | `35.112 -> 35.112` | `34.894 -> 34.894` | `688.86 -> 688.86` (`+0.00%`) | `693.17 -> 693.17` (`+0.00%`) | **`+0.00%`** (`+0.00%..+0.00%`) |
| `c8_med_accept_32k` | `40 -> 32` | `36.555 -> 31.552` | `36.340 -> 31.540` | `659.06 -> 694.80` (`+5.42%`) | `662.95 -> 695.07` (`+4.85%`) | **`+5.14%`** (`+4.85%..+5.42%`) |
| `c8_med_accept_64k` | `40 -> 32` | `37.378 -> 32.208` | `37.167 -> 32.210` | `646.26 -> 680.66` (`+5.32%`) | `649.93 -> 680.62` (`+4.72%`) | **`+5.02%`** (`+4.72%..+5.32%`) |
| `c8_med_accept_128k` | `40 -> 32` | `39.022 -> 33.538` | `38.819 -> 33.548` | `619.84 -> 653.67` (`+5.46%`) | `623.08 -> 653.47` (`+4.88%`) | **`+5.17%`** (`+4.88%..+5.46%`) |
| `c8_low_accept_64k` | `24 -> 16` | `29.946 -> 26.310` | `30.095 -> 26.273` | `492.41 -> 492.58` (`+0.03%`) | `489.97 -> 493.28` (`+0.68%`) | **`+0.36%`** (`+0.03%..+0.68%`) |
| `c8_low_accept_128k` | `24 -> 16` | `30.887 -> 26.964` | `31.070 -> 26.917` | `477.41 -> 480.65` (`+0.68%`) | `474.60 -> 481.47` (`+1.45%`) | **`+1.07%`** (`+0.68%..+1.45%`) |
| `c8_high_accept_64k` | `48 -> 48` | `39.711 -> 39.711` | `39.773 -> 39.773` | `933.94 -> 933.94` (`+0.00%`) | `932.48 -> 932.48` (`+0.00%`) | **`+0.00%`** (`+0.00%..+0.00%`) |
| `c16_second_wave_saturation_2k` | `80 -> 64` | `51.611 -> 40.720` | `51.986 -> 40.807` | `1206.26 -> 1351.28` (`+12.02%`) | `1197.56 -> 1348.38` (`+12.59%`) | **`+12.31%`** (`+12.02%..+12.59%`) |
| `c16_second_wave_saturation_64k` | `80 -> 64` | `56.044 -> 44.373` | `56.440 -> 44.433` | `1117.08 -> 1240.03` (`+11.01%`) | `1109.23 -> 1238.37` (`+11.64%`) | **`+11.33%`** (`+11.01%..+11.64%`) |
| `c8_heterogeneous_2k_128k` | `36 -> 32` | `35.805 -> 31.586` | `35.593 -> 31.582` | `653.14 -> 700.87` (`+7.31%`) | `657.03 -> 700.95` (`+6.68%`) | **`+7.00%`** (`+6.68%..+7.31%`) |

## Exact Commands & Verification

### 1. 2-Boot GPU Unit Test Suite (`sp8`)
```bash
docker run --rm --gpus all --ipc=host --entrypoint pytest \
  -v /tmp/vllm-graph-context-cost/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:ro \
  -v /tmp/vllm-graph-context-cost/tests/v1/spec_decode/test_adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_adaptive_verification.py:ro \
  spark-ds41f:1107 --noconftest -v /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_adaptive_verification.py
```
- **Boot 1 (`receipts/gpu-pytest-boot1.txt`)**: `44 passed, 1 skipped in 7.98s`
- **Boot 2 (`receipts/gpu-pytest-boot2.txt`)**: `44 passed, 1 skipped in 8.23s`

### 2. 2-Boot Hardware Kernel Profile & Screening Benchmark (`sp5` / `sp8`)
```bash
flock /tmp/gpu.lock -c '
docker run --rm --gpus all --ipc=host --entrypoint python3 \
  -v /home/chris/projects/spark-ds41f/cache:/cache:rw \
  -v /tmp/adaptive_verification_cand.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:ro \
  -v /tmp:/host_tmp:rw \
  -e TILELANG_CACHE_DIR=/cache/spark-ds41f/jit/tilelang \
  spark-ds41f:1107 /host_tmp/screen_graph_context_cost.py \
    --boot 1 \
    --output /host_tmp/screen-graph-context-cost-boot1.jsonl \
    --profile-json /host_tmp/attn-idx-profile-boot1.json
'
```

## Errors & Observations

1. **Packed ABI constraint during standalone MXFP8 GEMM profiling**: Initial standalone `gemm_mxfp8` microbenchmark calls failed with `RuntimeError: kernel gemm_mxfp8 input SFB shape[0] violates packed ABI constraint` until weights were formatted with `pack_weight_tiles` for `TILE_CONTIGUOUS_FP8` shapes and scales with `pack_scale_words`.
2. **Mid-tile vs tile-boundary trimming in heterogeneous batches**: Dividing `valid_scores / valid_scales` initially over-weighted the per-row context delta relative to the shared batch GEMM cost, trading high-survival `128K` drafts for much lower-survival `2K` drafts inside the same `decode_64` tile (`-1.2%` on `c8_heterogeneous_2k_128k`). Replacing multiplicative scaling in slot ordering with additive marginal surplus `valid_scores - row_cost * max(valid_scales - 1.0, 0.0)` and pricing the `R = 32 -> 33` `decode_32 -> decode_64` jump (`3 * mma_step`) aligned heterogeneous trimming with the `32`-row tile boundary (`+7.00%` 2-boot mean speedup).
3. **Memory Observations**: Pure CPU NumPy scalar/vector math on existing `req_states.num_computed_tokens_np` and `survival_probability` arrays (`<= 64` requests); zero additional GPU tensor allocations, zero CUDA graph pool impact, and zero change to KV cache footprint.

## Conclusion

Pricing the 16-row MMA tile transitions (`R > 16`, `R > 32`), the 64-row second-wave saturation knee (`R > 64`), and the two-slope long-context indexer/attention cost band (`8K..128K`) in `AdaptiveVerificationManager._step_cost` eliminates `R = 16 -> 20` single-request tile spills (`+1.12%`), curbs `R = 64 -> 80` second-wave over-admission (`+11.33%..+12.31%`), and trims low-yield long-context verification rows at `32K..128K` (`+5.02%..+5.17%` homogeneous, `+7.00%` heterogeneous `2K+128K`) with zero regression on short-context (`2K`) or high-acceptance regimes.
