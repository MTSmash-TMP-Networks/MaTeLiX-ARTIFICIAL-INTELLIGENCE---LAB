import unittest
import pickle
import tempfile
from pathlib import Path

try:
    import torch
    from matelix_ddp_worker import (
        DataCollator,
        DistContext,
        PlannedBatchIterableDataset,
        TokenizedShardIterableDataset,
        TrainConfig,
        audit_dataset,
        build_token_batch_plan,
        iter_accumulation_batches,
    )
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

    def test_token_batch_plan_is_ddp_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            samples = []
            for index in range(10):
                samples.append({
                    "input_ids": [1, 2], "attention_mask": [1, 1],
                    "labels": [-100, 2], "seq_len": 2,
                    "split_key": f"sample:{index}",
                })
            with open(cache_dir / "shard_000000.pkl", "wb") as handle:
                pickle.dump({
                    "shard_idx": 0, "global_start": 0,
                    "num_samples": len(samples), "samples": samples,
                }, handle)

            cfg = TrainConfig(
                model_dir="model", csv_path="data.csv", val_split=0,
                max_seq_length=8, per_device_train_batch_size=2,
                gradient_accumulation_steps=2, dynamic_token_batching=True,
                max_tokens_per_batch=16, max_samples_per_batch=8,
            )
            cfg.normalize()
            ctx = DistContext(0, 0, 4, True, torch.device("cpu"))
            batches, stats = build_token_batch_plan(cache_dir, cfg, ctx)
            self.assertEqual(len(batches) % 8, 0)
            self.assertEqual(stats["batches_per_rank"] % 2, 0)
            self.assertGreater(stats["ddp_padding_batches"], 0)

            dataset = PlannedBatchIterableDataset(
                cache_dir, batches, rank=0, world_size=4, seed=42, shuffle=False,
                original_batch_count=stats["original_batches"],
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, collate_fn=DataCollator(pad_token_id=0),
            )
            first_batch = next(iter(loader))
            self.assertEqual(first_batch["input_ids"].ndim, 2)
            self.assertLessEqual(first_batch["input_ids"].numel(), 16)

    def test_token_normalized_accumulation_weights(self):
        cfg = TrainConfig(
            model_dir="model", csv_path="data.csv",
            gradient_accumulation_steps=2, token_normalized_loss=True,
        )
        cfg.normalize()
        batches = [
            {"labels": torch.tensor([[-100, 1, -100]])},
            {"labels": torch.tensor([[-100, 1, 2]])},
        ]
        ctx = DistContext(0, 0, 1, False, torch.device("cpu"))
        window = list(iter_accumulation_batches(batches, cfg, ctx))
        self.assertAlmostEqual(window[0][1], 1 / 3)
        self.assertAlmostEqual(window[1][1], 2 / 3)
        self.assertTrue(window[-1][2])

    def test_dataset_audit_detects_thread_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "threads.csv"
            csv_path.write_text(
                "id,parent_id,Benutzer,Assistentin\n"
                "1,2,Frage 1,Antwort 1\n"
                "2,1,Frage 2,Antwort 2\n",
                encoding="utf-8",
            )
            cfg = TrainConfig(
                model_dir="model", csv_path=str(csv_path), template_mode="chat",
            )
            cfg.normalize()
            report = audit_dataset(cfg, root, is_main=True)
            self.assertFalse(report["ok"])
            self.assertEqual(report["thread_cycles"], 1)
            self.assertTrue((root / "dataset_audit.json").is_file())


if __name__ == "__main__":
    unittest.main()
