# Browser Model Performance — D1 Real-Chrome CLIP Optimization Study

**Measurement date:** October 20, 2026
**Measured browser:** Headless Chrome `155.0.8059.80` on the local Windows host
**Runtime:** ONNX Runtime Web `1.24.2`, Vite preview, and Playwright only as the browser driver
**Browser capability:** WebGPU was available; `navigator.hardwareConcurrency` was `16`; the page was not cross-origin isolated, so WASM used one thread.

## Scope and reproducibility

This D1 study replaces the earlier small, differently shaped timing samples in this document. It completed **51 actual Chrome inference runs** with no failed run and no page error. The raw report is `d1-browser-runtime/browser-results.json`; the grouped medians are in `d1-browser-runtime/summary.json`; `print_d1_browser_table.py` renders the same compact comparison table.

The fixture contains 68 directed relationship candidates made from nine recorded YOLO-World detections in `datasets/visual_genome/VG_100K_2/1.jpg`. Its pair admission rule is `i != j` and unequal detector labels. Reverse-directed pairs reuse the same union crop, leaving **26 unique preprocessed CLIP tensors**. Input tensors, geometry, subject IDs, object IDs, and the Python FP32 feature baseline were saved before browser execution.

Consequently, this study measures real browser model execution while deliberately excluding detector execution and browser image preprocessing. Each browser-produced CLIP feature matrix is passed to the unchanged relationship MLP with the recorded geometry and label IDs. This isolates provider, model-format, batch/chunk, and duplicate-crop effects without conflating them with Pillow/OpenCLIP crop-and-resize parity.

## Model formats and session behavior

| Format | Model size | Browser result in the completed 51-run data set |
|---|---:|---|
| FP32 | 351,780,080 bytes (335.5 MiB) | Measured with WASM and WebGPU. |
| INT8 dynamic per-channel | 89,350,891 bytes (85.2 MiB) | Measured with WASM and WebGPU. |
| FP16, I/O kept FP32 | 176,081,784 bytes after repair (168.0 MiB) | **Not measured in this result set.** |

Sessions were cached once per `(provider, format)` rather than recreated for each repetition. This keeps the timing rows focused on `session.run()` and records cold session construction separately:

| Provider / format | First session load |
|---|---:|
| WASM FP32 | 69,204.0 ms |
| WASM INT8 | 7,563.9 ms |
| WebGPU FP32 | 20,469.1 ms |
| WebGPU INT8 | 6,255.4 ms |

The FP16 omission is an artifact-status issue, not evidence that Chrome/WebGPU lacks FP16 support. The original FP16 conversion failed ONNX topological validation (`graph_input_cast_0` was consumed before it was produced). During post-benchmark investigation, `try_fp16_repair.py` produced checker-valid models by topologically sorting the converted graph. The repaired file was not published to the benchmark URL or rerun in Chrome, so no FP16 latency or parity result is claimed here. In the completed benchmark, the absent public FP16 URL returned Vite's HTML shell and was correctly skipped.

## Median warm-run performance

Each WASM configuration was repeated three times; each measured WebGPU FP32 configuration twice; WebGPU INT8 once. Times are medians where repetitions exist and include CLIP inference, duplicate reconstruction, and relationship-MLP inference. Session creation is excluded.

| Provider | Format | Candidate mode | CLIP chunk | Runs | CLIP | Reconstruct | MLP | Total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| WASM | FP32 | directed (68 tensors) | 1 | 3 | 63,678.7 ms | 0.2 ms | 14.2 ms | 63,683.0 ms |
| WASM | FP32 | directed | 4 | 3 | 53,708.9 ms | 0.0 ms | 5.9 ms | 53,720.9 ms |
| WASM | FP32 | directed | 8 | 3 | 48,723.6 ms | 0.1 ms | 7.7 ms | 48,736.6 ms |
| WASM | FP32 | directed | 16 | 3 | 31,236.8 ms | 0.0 ms | 2.5 ms | 31,239.3 ms |
| WASM | FP32 | directed | 32 | 3 | 28,814.3 ms | 0.0 ms | 2.5 ms | 28,819.9 ms |
| WASM | FP32 | directed | 68 | 3 | 28,599.7 ms | 0.0 ms | 2.6 ms | 28,601.3 ms |
| WASM | FP32 | deduplicated (26 tensors) | 1 | 3 | 9,970.0 ms | 0.2 ms | 3.1 ms | 9,973.3 ms |
| WASM | FP32 | deduplicated | 4 | 3 | 10,053.1 ms | 0.1 ms | 1.9 ms | 10,055.1 ms |
| WASM | FP32 | deduplicated | 8 | 3 | **9,921.1 ms** | 0.1 ms | 1.2 ms | **9,922.3 ms** |
| WASM | FP32 | deduplicated | 16 | 3 | 10,142.0 ms | 0.0 ms | 1.8 ms | 10,143.7 ms |
| WASM | FP32 | deduplicated | 32 | 3 | 12,510.9 ms | 0.3 ms | 2.5 ms | 12,513.8 ms |
| WASM | FP32 | deduplicated | 34 | 3 | 10,836.0 ms | 0.2 ms | 2.1 ms | 10,837.8 ms |
| WASM | INT8 | directed | 68 | 3 | 24,287.6 ms | 0.0 ms | 1.8 ms | 24,289.4 ms |
| WASM | INT8 | deduplicated | 34 | 3 | 8,716.2 ms | 0.2 ms | 1.3 ms | 8,720.8 ms |
| WebGPU | FP32 | directed | 1 | 2 | 13,647.1 ms | 0.0 ms | 7.0 ms | 13,654.1 ms |
| WebGPU | FP32 | directed | 8 | 2 | **7,899.1 ms** | 0.0 ms | 3.6 ms | **7,902.7 ms** |
| WebGPU | FP32 | directed | 68 | 2 | 10,020.9 ms | 0.0 ms | 7.2 ms | 10,028.1 ms |
| WebGPU | FP32 | deduplicated | 34 | 2 | **3,932.1 ms** | 0.2 ms | 5.8 ms | **3,938.1 ms** |
| WebGPU | INT8 | deduplicated | 34 | 1 | 18,738.9 ms | 0.3 ms | 8.1 ms | 18,747.3 ms |

### Performance findings

- Deduplicating exact reverse-direction crop tensors is the largest safe optimization in this fixture: WASM FP32 falls from 28,601.3 ms for directed chunk 68 to 9,922.3 ms for deduplicated chunk 8 — a **65.3%** end-to-end reduction. Reconstruction costs at most 0.3 ms.
- For directed FP32 work, large WASM chunks are best in this measurement: chunk 68 is 28,601.3 ms, versus 48,736.6 ms at chunk 8 and 63,683.0 ms at chunk 1.
- WebGPU executes the model correctly and is materially faster for this workload. Its best directed FP32 row is 7,902.7 ms (chunk 8). Its deduplicated FP32 row is 3,938.1 ms, **60.3% faster** than the best WASM-deduplicated FP32 row.
- INT8 reduces download size by about 74.6% and modestly improves WASM runtime (24,289.4 ms directed, 8,720.8 ms deduplicated), but it changes downstream decisions. WebGPU INT8 is substantially slower than WebGPU FP32 in this run and is not a performance recommendation.
- The relationship MLP is consistently a millisecond-scale tail. CLIP dominates total time; it is the optimization target.

## Numerical parity and decision preservation

Feature parity compares browser CLIP output with the saved Python FP32 features. Probability parity compares each browser run with the first successful browser WASM FP32-directed probability baseline. `class changes` counts different argmax predicate IDs over 68 candidates; `retained changes` counts changes to the production-style predicate filter (`class_id != 50` and confidence >= 0.30).

| Provider / format | Feature max abs. error | Feature cosine | Probability max abs. error | Probability cosine | Class changes | Retained changes | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| WASM FP32 | `1.678e-05` | `0.999999999996` | `0` | `1.0` | 0 | 0 | Exact downstream baseline agreement across chunk/mode rows. |
| WebGPU FP32 | `2.289e-05` | `0.999999999993` | `1.773e-06` | effectively `1.0` | 0 | 0 | Numerically close and all 68 downstream decisions preserved. |
| WASM INT8 | `1.364` | `0.942229` | `0.2614` | `0.991251` | 6 | 6 | Faster/smaller, but changes predictions and retained relationships. |
| WebGPU INT8 | `1.314` | `0.942685` | `0.2399` | `0.991194` | 6 | 6 | Also changes predictions and is slower than WebGPU FP32. |

Python CPU reference results reinforce that INT8 must not be treated as a drop-in replacement: its 68-pair FP32 comparison recorded feature cosine `0.420900`, probability cosine `0.978664`, nine predicate-class changes, and nine retention changes. The exact error differs by runtime, but both Python and browser results show nonzero decision drift. FP32 preserves the modeled decisions.

## Recommendation

| Component / choice | Conclusion |
|---|---|
| FP32 visual CLIP + WebGPU | **Recommended browser path for correctness and measured speed.** Use exact-crop deduplication and a bounded scheduler; the measured 26-crop result is 3.94 s warm, plus a 20.47 s first WebGPU session build. |
| FP32 visual CLIP + WASM | **Correct fallback, not interactive at full workload.** It preserves results but needs about 9.92 s warm with deduplication and about 69.20 s for the initial FP32 session on this host. |
| INT8 visual CLIP | **Do not enable as a transparent optimization.** It needs task-level acceptance criteria or retraining/calibration because it changes six of the 68 browser predictions and retention outcomes. |
| FP16 visual CLIP | **Deferred.** A checker-valid conversion now exists, but must be published and measured in Chrome before any recommendation. |
| Relationship MLP | **Browser-ready.** Its cost is negligible relative to CLIP and the existing FP32 graph can remain unchanged. |
| Full browser-only Scene Graph Explorer | **Technically valid but not yet interactive on the measured host.** Even the best verified FP32 path is seconds per fixture, before browser preprocessing and YOLO detection. |

## Limits and next work

1. Establish end-to-end browser preprocessing parity for expanded union crops, Python rounding, Pillow/OpenCLIP bicubic resize, center crop, RGB conversion, and normalization.
2. Publish the repaired FP16 artifact, run the same 51-run matrix (or a documented subset), and compare its decisions against FP32 before considering it.
3. Profile WebGPU provider partitioning and memory behavior; ONNX Runtime emitted the usual notice that some shape-related nodes may remain on CPU.
4. Measure persistent HTTP/model caching separately. This experiment intentionally used `cache: 'no-store'` so session-load timings are transparent.
5. Repeat on user-facing Chrome and Microsoft Edge hardware; these results are one local Windows host and should not be generalized as device-independent latency.

No production integration, model substitution, retraining, frontend migration, or deployment change was performed by this study.
