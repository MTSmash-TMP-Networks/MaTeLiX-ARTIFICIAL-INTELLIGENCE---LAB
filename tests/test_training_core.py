import unittest

try:
    from matelix_ddp_worker import TokenizedShardIterableDataset, TrainConfig
except ModuleNotFoundError as exc:  # Allows source-only environments without PyTorch.
    TokenizedShardIterableDataset = TrainConfig = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"training dependencies unavailable: {IMPORT_ERROR}")
class TrainingCoreTests(unittest.TestCase):
    def test_split_is_stable_and_exclusive(self):
        train = TokenizedShardIterableDataset(".", 0, 1, split="train", val_split=0.2, split_seed=42)
        val = TokenizedShardIterableDataset(".", 0, 1, split="validation", val_split=0.2, split_seed=42)
        for index in range(1000):
            key = f"thread:{index // 3}"
            self.assertNotEqual(train._belongs_to_split(index, key), val._belongs_to_split(index, key))

    def test_whole_thread_stays_in_one_split(self):
        dataset = TokenizedShardIterableDataset(".", 0, 1, split="validation", val_split=0.2, split_seed=9)
        decisions = {dataset._belongs_to_split(i, "thread:abc") for i in range(50)}
        self.assertEqual(len(decisions), 1)

    def test_config_normalization(self):
        cfg = TrainConfig(
            model_dir="model", csv_path="data.csv", val_split=0.9,
            early_stopping_patience=-1, lora_dropout=2.0,
            lora_target_modules=[" q_proj ", "q_proj", "v_proj"],
        )
        cfg.normalize()
        self.assertEqual(cfg.val_split, 0.5)
        self.assertEqual(cfg.early_stopping_patience, 0)
        self.assertEqual(cfg.lora_dropout, 0.95)
        self.assertEqual(cfg.lora_target_modules, ["q_proj", "v_proj"])


if __name__ == "__main__":
    unittest.main()
