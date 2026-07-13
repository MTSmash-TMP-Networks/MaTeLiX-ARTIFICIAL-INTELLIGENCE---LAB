# Changelog

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
- optional mixed training from dialog fields and an additional `Text` column

### Changed

- V100-oriented UI defaults disable unnecessary checkpointing and cache clearing
- exact batch-plan step counts replace projected adaptive scheduler counts
- incomplete DDP accumulation tails are padded deterministically instead of discarded
- validation and training metrics use causal-shifted target-token weighting
- batch planning and dataset auditing run once on rank 0 to reduce startup I/O
- token telemetry reports the exact all-rank input-token count
- FP16 gradient overflows no longer advance optimizer or scheduler step counters
- DDP defaults to `find_unused_parameters=false` to avoid redundant graph traversal
- mixed dialog/text samples from the same thread remain in the same data split

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
