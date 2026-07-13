# Changelog

## 8.3.0 - 2026-07-13

### Added

- optional mixed training from dialog fields and an additional `Text` column
- tokenizer-aware long-text chunking with configurable overlap and stable document splits
- EOS termination for every free-text document and chunk
- explicit `pretrain`, `mixed`, `sft`, and backward-compatible `custom` training phases
- token-based dialog/text mixture weights with bounded deterministic oversampling
- normalized exact deduplication plus optional approximate near-duplicate detection
- quality checks for corrupted characters, HTML boilerplate, repetition, and short texts
- optional split-safe short-text packing with cross-document target masking
- optional scratch-tokenizer training from the active corpus
- separate text and dialog validation metrics
- detailed tokenization reports with skip reasons, type counts, token counts, chunks, packing, and length percentiles

### Changed

- scratch pretraining automatically enables full prompt loss and a safe warmup when no warmup was configured
- scratch position embeddings are expanded to cover the configured context length
- training phases control which dialog/text sources feed both the model and N-gram tokenizer extension
- normalized duplicates can be removed before tokenization instead of only being reported
- packed samples receive a fixed train/validation assignment before packing to prevent split leakage
- retained near-duplicate samples inherit their representative's split to prevent validation leakage
- rank-0 setup failures are propagated to every DDP worker instead of waiting for the collective timeout
- live loss and learning-rate charts now share one optimizer-step timeline and render side by side

## 8.2.0 - 2026-07-13

### Added

- dynamic token-budget batching with automatic safe defaults
- deterministic global batch plans aligned to DDP world size and gradient accumulation
- token-normalized loss accumulation across all ranks
- automatic DataLoader worker selection and configurable prefetching
- optional NEFTune embedding noise
- pre-training CSV audit with strict mode and JSON report
- hardware profile reporting with Volta/V100 capability detection
- live batch-plan, padding-efficiency and worker metrics

### Changed

- V100-oriented UI defaults disable unnecessary checkpointing and cache clearing
- exact batch-plan step counts replace projected adaptive scheduler counts
- incomplete DDP accumulation tails are padded deterministically instead of discarded
- validation and training metrics use causal-shifted target-token weighting
- batch planning and dataset auditing run once on rank 0 to reduce startup I/O
- token telemetry reports the exact all-rank input-token count
- FP16 gradient overflows no longer advance optimizer or scheduler step counters
- DDP defaults to `find_unused_parameters=false` to avoid redundant graph traversal

## 8.1.0 - 2026-07-13

### Added

- resumable Full-FT and LoRA checkpoints with optimizer, scheduler, scaler and per-rank RNG state
- deterministic, conversation-grouped train/validation splitting
- distributed token-weighted validation loss and perplexity
- early stopping with configurable patience and minimum delta
- automatic `best_model` artifacts
- configurable LoRA dropout and target modules
- validation, checkpoint and resume controls in the Web UI
- pinned runtime dependency ranges and GitHub Actions CI

### Fixed

- training options exposed by the API are no longer silently discarded by the worker
- scheduler step estimates now exclude reserved validation samples
- resumed DDP workers restore their own random-number-generator state
