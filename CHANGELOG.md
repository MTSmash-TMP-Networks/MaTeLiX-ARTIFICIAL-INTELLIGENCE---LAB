# Changelog

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
