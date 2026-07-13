import unittest
import pickle
import re
import tempfile
from pathlib import Path

from matelix_ngram_pipeline import iter_training_texts

try:
    import torch
    from matelix_ddp_worker import (
        DataCollator,
        DistContext,
        FixedLRScheduler,
        NearDuplicateTracker,
        PlainTextSample,
        PlannedBatchIterableDataset,
        SampleRef,
        ShortTextPacker,
        StructuredChatSample,
        TokenizedShardIterableDataset,
        TrainConfig,
        audit_dataset,
        apply_token_mixture_weights,
        build_examples_stream,
        build_token_batch_plan,
        count_examples_fast,
        iter_accumulation_batches,
        tokenize_text_examples,
    )
except ModuleNotFoundError as exc:  # Allows source-only environments without PyTorch.
    TokenizedShardIterableDataset = TrainConfig = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class NgramTrainingTextTests(unittest.TestCase):
    def test_mixed_texts_are_included_and_cycles_terminate(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mixed-cycle.csv"
            csv_path.write_text(
                "id,parent_id,system,Benutzer,Kontext,Assistentin,Text\n"
                "1,2,System,Frage 1,,Antwort 1,Freier Text 1\n"
                "2,1,,Frage 2,,Antwort 2,Freier Text 2\n",
                encoding="utf-8",
            )

            dialog_only = list(iter_training_texts(
                str(csv_path), "dialogplus", "text",
            ))
            mixed = list(iter_training_texts(
                str(csv_path), "dialogplus", "text", True, "Text",
            ))

            self.assertEqual(len(dialog_only), 2)
            self.assertEqual(mixed[:2], dialog_only)
            self.assertEqual(mixed[2:], ["Freier Text 1", "Freier Text 2"])

            pretrain = list(iter_training_texts(
                str(csv_path), "dialogplus", "text", True, "Text", "pretrain",
            ))
            sft = list(iter_training_texts(
                str(csv_path), "dialogplus", "text", True, "Text", "sft",
            ))
            self.assertEqual(pretrain, ["Freier Text 1", "Freier Text 2"])
            self.assertEqual(sft, dialog_only)


@unittest.skipIf(IMPORT_ERROR is not None, f"training dependencies unavailable: {IMPORT_ERROR}")
class TrainingCoreTests(unittest.TestCase):
    class WhitespaceTokenizer:
        eos_token_id = 99
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            matches = list(re.finditer(r"\S+", text))
            payload = {"input_ids": list(range(1, len(matches) + 1))}
            if return_offsets_mapping:
                payload["offset_mapping"] = [(match.start(), match.end()) for match in matches]
            return payload

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

        scratch_cfg = TrainConfig(
            model_dir="model", csv_path="data.csv", train_from_scratch=True,
            training_phase="pretrain", warmup_steps=0, warmup_ratio=0,
        )
        scratch_cfg.normalize()
        self.assertTrue(scratch_cfg.include_prompt_loss)
        self.assertEqual(scratch_cfg.warmup_ratio, 0.02)

    def test_fixed_scheduler_uses_exact_plan_and_loads_legacy_state(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = FixedLRScheduler(
            optimizer,
            base_lr=0.1,
            schedule="linear",
            total_steps=10,
            warmup_steps=2,
            min_lr_ratio=0.1,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        self.assertAlmostEqual(scheduler.step(2), 0.1)
        self.assertAlmostEqual(scheduler.step(10), 0.01)
        self.assertEqual(scheduler.state_dict()["scheduler_type"], "fixed_batch_plan")

        legacy_state = {
            **scheduler.state_dict(),
            "adaptive_enabled": True,
            "freeze_on_done": True,
            "never_increase_lr": True,
            "only_extend_steps": True,
        }
        restored = FixedLRScheduler(
            optimizer,
            base_lr=0.2,
            schedule="cosine",
            total_steps=3,
        )
        restored.load_state_dict(legacy_state)
        self.assertEqual(restored.total_steps, 10)
        self.assertEqual(restored.schedule, "linear")
        self.assertFalse(hasattr(restored, "adaptive_enabled"))

    def test_long_text_is_chunked_with_overlap_and_eos(self):
        tokenizer = self.WhitespaceTokenizer()
        item = PlainTextSample(
            text="Eins zwei drei vier. Fünf sechs sieben acht. Neun zehn elf zwölf.",
            split_key="thread:doc",
        )
        samples = tokenize_text_examples(
            item,
            tokenizer,
            6,
            chunk_long_texts=True,
            text_chunk_overlap=2,
            text_chunk_min_tokens=2,
            append_eos_to_text=True,
        )
        self.assertGreater(len(samples), 1)
        self.assertTrue(all(len(sample["input_ids"]) <= 6 for sample in samples))
        self.assertTrue(all(sample["input_ids"][-1] == 99 for sample in samples))
        self.assertTrue(all(sample["split_key"] == "thread:doc" for sample in samples))
        self.assertTrue(all(sample["chunk_count"] == len(samples) for sample in samples))

    def test_short_text_packing_never_crosses_dataset_split(self):
        def sample(split, key, token):
            return {
                "input_ids": [token, 99], "attention_mask": [1, 1],
                "labels": [token, 99], "seq_len": 2,
                "split_key": key, "assigned_split": split,
                "sample_type": "text", "original_token_count": 1,
            }

        packer = ShortTextPacker(8)
        outputs = []
        outputs.extend(packer.add(sample("train", "a", 1)))
        outputs.extend(packer.add(sample("validation", "b", 2)))
        outputs.extend(packer.add(sample("train", "c", 3)))
        outputs.extend(packer.flush())

        self.assertEqual({item["assigned_split"] for item in outputs}, {"train", "validation"})
        train = next(item for item in outputs if item["assigned_split"] == "train")
        self.assertEqual(train["packed_segment_count"], 2)
        self.assertEqual(train["labels"][2], -100)

    def test_token_mixture_weights_are_based_on_tokens(self):
        cfg = TrainConfig(
            model_dir="model", csv_path="data.csv", mixed_training=True,
            training_phase="mixed", text_token_weight=0.75,
            dialog_token_weight=0.25, max_mixture_oversample=4,
        )
        cfg.normalize()
        refs = [
            SampleRef(0, 0, 0, 100, "text"),
            SampleRef(0, 1, 1, 100, "dialog"),
        ]
        weighted, stats = apply_token_mixture_weights(refs, cfg)
        self.assertTrue(stats["enabled"])
        self.assertEqual(sum(ref.seq_len for ref in weighted if ref.sample_type == "text"), 300)
        self.assertEqual(sum(ref.seq_len for ref in weighted if ref.sample_type == "dialog"), 100)

    def test_near_duplicates_reuse_representative_split_key(self):
        tracker = NearDuplicateTracker(0.92)
        first_is_near, first_key = tracker.add_and_find(
            "Ein Router verbindet mehrere Netzwerke miteinander.", "thread:first",
        )
        second_is_near, second_key = tracker.add_and_find(
            "Ein Router verbindet mehrere Netzwerke miteinander.", "thread:second",
        )
        self.assertFalse(first_is_near)
        self.assertIsNone(first_key)
        self.assertTrue(second_is_near)
        self.assertEqual(second_key, "thread:first")

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

    def test_optional_mixed_dialog_and_text_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mixed.csv"
            csv_path.write_text(
                "id,parent_id,system,Benutzer,Kontext,Assistentin,Text\n"
                "1,,System,Hallo,,Antwort 1,Freier Text 1\n"
                "2,1,,Frage,,Antwort 2,\n"
                "3,,,,,,Nur freier Text\n"
                "4,,System,Andere Frage,,Andere Antwort,Freier Text 1\n",
                encoding="utf-8",
            )
            cfg = TrainConfig(
                model_dir="model",
                csv_path=str(csv_path),
                template_mode="dialogplus",
                mixed_training=True,
                mixed_text_column="Text",
            )
            cfg.normalize()

            samples = list(build_examples_stream(cfg))
            self.assertEqual(len(samples), 6)
            self.assertIsInstance(samples[0], StructuredChatSample)
            self.assertIsInstance(samples[1], PlainTextSample)
            self.assertEqual(samples[0].split_key, samples[1].split_key)
            self.assertEqual(samples[0].split_key, samples[4].split_key)
            self.assertEqual(samples[1].split_key, samples[5].split_key)
            self.assertEqual(count_examples_fast(cfg), 6)

            report = audit_dataset(cfg, root, is_main=True)
            self.assertTrue(report["mixed_training_ready"])
            self.assertEqual(report["usable_assistant_samples"], 3)
            self.assertEqual(report["usable_mixed_text_samples"], 3)
            self.assertEqual(report["duplicate_mixed_text_samples"], 1)

            cfg.mixed_training = False
            dialog_only = list(build_examples_stream(cfg))
            self.assertEqual(len(dialog_only), 3)
            self.assertTrue(all(isinstance(sample, StructuredChatSample) for sample in dialog_only))

            cfg.mixed_training = True
            cfg.mixed_text_column = "Text"
            cfg.training_phase = "pretrain"
            pretrain_only = list(build_examples_stream(cfg))
            self.assertEqual(len(pretrain_only), 3)
            self.assertTrue(all(isinstance(sample, PlainTextSample) for sample in pretrain_only))

            cfg.training_phase = "sft"
            sft_only = list(build_examples_stream(cfg))
            self.assertEqual(len(sft_only), 3)
            self.assertTrue(all(isinstance(sample, StructuredChatSample) for sample in sft_only))

            cfg.mixed_training = True
            cfg.mixed_text_column = "Fehlt"
            cfg.training_phase = "custom"
            with self.assertRaisesRegex(ValueError, "CSV-Spalten fehlen: Fehlt"):
                build_examples_stream(cfg)


if __name__ == "__main__":
    unittest.main()
