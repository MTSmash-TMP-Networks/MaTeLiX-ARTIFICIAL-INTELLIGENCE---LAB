# MaTeLiX ARTIFICIAL INTELLIGENCE - LAB

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Web%20API-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Training-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

<img width="3024" height="2206" alt="Screenshot 2026-05-02 at 19-34-15 MaTeLiX AI – Training Dashboard" src="https://github.com/user-attachments/assets/6a7539df-8837-441a-93e6-bed1c1bc9c53" />


Local **LLM training and inference lab** with **FastAPI**, **Web UI**, **DDP / Multi-GPU training**, **LoRA**, **live logs**, **live preview** and an **OpenAI-compatible API**.

> Built for local fine-tuning, chat datasets, text datasets and controlled inference workflows.



## Aktuelle Version

**Stand: Version 8.3**

Enthalten sind unter anderem:

- token-budget-gesteuertes Whole-Turn-Packing
- DDP / Multi-GPU-Training
- LoRA-Training mit optionalem Merge
- resumierbare Epoch-Checkpoints inklusive Optimizer, Scheduler, AMP-Scaler und RNG-Zuständen
- konfigurierbare LoRA-Zielmodule und LoRA-Dropout
- deterministischer Train/Validation-Split ohne Datenüberschneidung
- DDP-synchronisierte Validation Loss und Perplexity
- Early Stopping und automatische Speicherung von `best_model`
- dynamische Batches mit festem Tokenbudget statt starrer Samplezahl
- DDP-ausgerichteter Batchplan ohne verworfene Accumulation-Reste
- global tokennormalisierte Gradient Accumulation
- automatische DataLoader-Worker und Prefetching
- optionales NEFTune und Dataset-Audit vor dem Modellstart
- Hardware-Profilierung mit V100-/Volta-Erkennung
- OpenAI-kompatible `/v1/*` API und Web-UI

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Start](#start)
- [Web UI](#web-ui)
- [Dataset Formats](#dataset-formats)
- [Strict Whole-Turn Packing](#strict-whole-turn-packing)
- [Training](#training)
- [Profil für 4× NVIDIA V100 32 GB](#profil-für-4-nvidia-v100-32-gb)
- [Inference](#inference)
- [OpenAI-compatible API](#openai-compatible-api)
- [Important Parameters](#important-parameters)
- [DDP / Multi-GPU](#ddp--multi-gpu)
- [LoRA / Merge Behavior](#lora--merge-behavior)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Trademark](#trademark)

---

## Overview

**MaTeLiX ARTIFICIAL INTELLIGENCE - LAB** is a local environment for:

- supervised fine-tuning of LLMs
- chat and text dataset processing
- LoRA or full training
- DDP / multi-GPU execution
- browser-based training control
- local inference and streaming chat
- OpenAI-style API access

It is designed for practical local experiments with reproducible outputs, cached tokenized datasets and a lightweight but powerful browser UI.

---

## Features

### Training
- Full fine-tuning or LoRA fine-tuning
- CPU, MPS, single-GPU or **DDP / multi-GPU**
- CSV-based datasets
- live training status
- stop mechanism for running jobs
- structured training outputs per run

### Dataset Processing
- supports:
  - `chat`
  - `dialogplus`
  - `plain`
- reconstructs thread chains via `id` / `parent_id`
- **strict whole-turn packing**
- no partial dialog turns
- oversized samples are skipped cleanly
- tokenized **shard cache** for faster re-runs

### Tokenizer / Templates
- automatically adds:
  - `<|System|>`
  - `<|Benutzer|>`
  - `<|Assistentin|>`
- auto pad-token handling if missing
- custom MaTeLiX chat template
- strict role validation in chat mode

### Inference
- load / unload model
- standard chat
- streaming chat
- base model or LoRA adapter loading
- prefers the latest available trained model

### UI / API
- FastAPI backend
- OpenAI-compatible `/v1/*` endpoints
- Web UI with:
  - hardware stats
  - training setup
  - logs
  - loss chart
  - live sample preview
  - browser chat

---

## Screenshots

### Dashboard / Training UI

![Dashboard Overview](docs/screenshots/dashboard-overview.png)

### Training Status / Logs

![Training Status](docs/screenshots/training-status.png)

### Browser Chat / Inference

![Browser Chat](docs/screenshots/browser-chat.png)

> Put your screenshots into `docs/screenshots/` using exactly these filenames, or adjust the paths above.

---

## Project Structure

```text
.
├─ matelix_lab_server_web_ddp.py
├─ matelix_ddp_worker.py
├─ matelix_ngram_pipeline.py
├─ requirements.txt
├─ tests/
│  └─ test_training_core.py
├─ .github/workflows/
│  └─ ci.yml
├─ datasets/
│  └─ *.csv
├─ static/
│  └─ index.html
├─ docs/
│  └─ screenshots/
│     ├─ dashboard-overview.png
│     ├─ training-status.png
│     └─ browser-chat.png
├─ training_outputs/
│  └─ <model>_YYYY-MM-DD_HH-MM-SS/
│     ├─ train_config.json
│     ├─ worker_config.json
│     ├─ training.log
│     ├─ status.json
│     ├─ livepreview.json
│     ├─ dataset_cache/
│     ├─ template_info.json
│     ├─ merged/
│     └─ ...
└─ README.md
````

---

## Installation

### Requirements

* Python 3.10+
* PyTorch
* `transformers`
* `fastapi`
* `uvicorn`
* `psutil`
* optional:

  * `peft` for LoRA support
  * CUDA for GPU / DDP training

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -U pip
# Zuerst den zu CUDA/ROCm/CPU passenden PyTorch-Build installieren:
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

> For CUDA, install the matching PyTorch build for your system.

---

## Start

```bash
python matelix_lab_server_web_ddp.py
```

Default URL:

```text
http://127.0.0.1:8002/
```

---

## Web UI

The browser UI is served from:

```text
/static/index.html
```

Main UI features:

* model selection
* dataset selection
* training configuration
* optional history cap
* cache rebuild toggle
* LoRA options
* live logs
* loss chart
* live sample preview
* browser-based inference chat

---

## Dataset Formats

## 1. `template_mode="plain"`

For plain text datasets.

Example:

```csv
text
Das ist ein Beispielsatz.
Noch ein Beispielsatz.
```

Typical config:

```json
{
  "template_mode": "plain",
  "column_name": "text"
}
```

---

## 2. `template_mode="chat"`

For threaded chat datasets.

Expected fields:

* `id`
* `parent_id`
* `system` (optional)
* `Benutzer`
* `Kontext` (optional)
* `Assistentin`

Example:

```csv
id,parent_id,system,Benutzer,Kontext,Assistentin
1,,Du bist ein freundlicher Chatbot.,Hallo!,,"Hallo! Wie kann ich dir helfen?"
2,1,,Wie spät ist es?,,"Ich habe keinen Zugriff auf deine Uhr, aber du kannst oben rechts schauen."
3,,Du bist Übersetzer.,Übersetze: "Guten Morgen",,Auf Englisch: "Good morning".
```

---

## 3. Gemischtes Dialog- und Texttraining

Das erweiterte Format kombiniert strukturierte Dialoge mit freien Texten:

```csv
id,parent_id,system,Benutzer,Kontext,Assistentin,Text
1,,Du bist ein freundlicher Chatbot.,Hallo!,,"Hallo! Wie kann ich dir helfen?",Eine freundliche Begrüßung eröffnet ein Gespräch.
2,1,,Was ist ein Router?,,"Ein Router verbindet unterschiedliche Netzwerke.",
3,,,,,,Ein Router leitet Datenpakete zwischen Netzwerken weiter.
```

Aktivierung:

```json
{
  "template_mode": "dialogplus",
  "mixed_training": true,
  "mixed_text_column": "Text"
}
```

Im Mischmodus kann eine Zeile einen Dialog, freien Text oder beides enthalten.
Ein nicht leeres Feld `Text` wird als eigener Plain-Text-Trainingssample genutzt.
Dialog und Text derselben ID beziehungsweise desselben Threads bleiben durch
denselben Split-Schlüssel gemeinsam im Train- oder Validation-Split. Ist der
gleiche freie Text in mehreren Threads vorhanden, werden auch diese Gruppen
gekoppelt, sodass exakte Textduplikate nicht über beide Splits verteilt werden.
Ist der Mischmodus ausgeschaltet, wird `Text` vollständig ignoriert und das bisherige
sechsspaltige Dialogformat verhält sich unverändert.

Die aktive Trainingsphase bestimmt, welche Spalten genutzt werden:

| Phase | Daten | Loss |
| --- | --- | --- |
| `pretrain` | freie Texte aus `Text` beziehungsweise Plain-CSV | alle Tokens |
| `mixed` | freie Texte und strukturierte Dialoge | bei Scratch alle Tokens, sonst konfigurierbar |
| `sft` | ausschließlich strukturierte Dialoge | nur Assistentinnen-Zielantwort |
| `custom` | bisheriges Verhalten | über `include_prompt_loss` steuerbar |

---

## 4. `template_mode="dialogplus"`

Works similar to `chat`, but uses a block-style conversation format.

Also applies:

* whole-turn packing
* no partial blocks
* oversized samples are skipped

---

## Strict Whole-Turn Packing

This version no longer relies on a fixed history window as the primary logic.

Instead it is **token-budget driven**:

* complete turns are collected from the end backwards
* only full blocks are included
* assistant target stays complete
* no cutting in the middle of a turn
* oversized samples are skipped

Freie Texte sind davon ausgenommen: Ist `chunk_long_texts=true`, werden sie
tokenizerbasiert an bevorzugten Absatz- oder Satzgrenzen geteilt. Alle Chunks
eines Dokuments behalten denselben Split-Schlüssel. Mit
`text_chunk_overlap=128` wird Kontext an den Grenzen wiederholt, ohne dass ein
Dokument gleichzeitig in Train und Validation gelangen kann. Jeder Text-Chunk
endet standardmäßig mit dem EOS-Token.

### Recommended behavior

For most chat datasets:

```json
{
  "max_history_turns": null
}
```

That means:

* only the token window decides
* no extra artificial turn cap

You can still set `max_history_turns` if you want an additional hard limit.

---

## Scratch-Trainingspipeline

Für ein zufällig initialisiertes Modell sollten die drei Phasen nacheinander
ausgeführt und jeweils vom letzten Checkpoint fortgesetzt werden.

### Phase 1: Grundtraining

```json
{
  "train_from_scratch": true,
  "train_mode": "full",
  "training_phase": "pretrain",
  "mixed_training": true,
  "chunk_long_texts": true,
  "text_chunk_overlap": 128,
  "append_eos_to_text": true,
  "warmup_ratio": 0.02
}
```

In dieser Phase wird der Loss über sämtliche Texttokens berechnet. Ist kein
Warmup gesetzt, verwendet Scratch-Pretraining automatisch zwei Prozent.

### Phase 2: Gemischtes Training

```json
{
  "training_phase": "mixed",
  "mixed_training": true,
  "text_token_weight": 0.7,
  "dialog_token_weight": 0.3,
  "max_mixture_oversample": 4.0
}
```

Die Gewichtung arbeitet nach Tokens statt nach Zeilen. Unterrepräsentierte
Sampletypen werden deterministisch und begrenzt mehrfach eingeplant; kein Original
wird für die Gewichtung verworfen, außer sein Zielgewicht wird ausdrücklich auf
`0` gesetzt.

### Phase 3: Instruction-SFT

```json
{
  "training_phase": "sft",
  "mixed_training": true,
  "include_prompt_loss": false
}
```

Die `Text`-Spalte wird in dieser Phase ignoriert. Trainiert wird nur die
Assistentinnen-Antwort, während System, Benutzer und Kontext als Prompt dienen.

### Optionaler Scratch-Tokenizer

Mit `train_scratch_tokenizer=true` trainiert der Worker vor dem Modellstart einen
neuen Fast-Tokenizer aus dem aktiven Korpus. Das Vokabular wird unter
`scratch_tokenizer/` im Run-Verzeichnis gespeichert und in allen DDP-Ranks sowie
im Shard-Producer identisch verwendet. Bei einem Resume wird immer der
Tokenizer des Checkpoints weiterverwendet.

Der Batch-Plan schreibt außerdem Modellparameter, unverfälschte Trainingstokens
pro Epoche und das geschätzte Verhältnis `Tokens/Parameter` in
`batch_plan.json`. Bei weniger als zehn geplanten Tokens pro Parameter warnt ein
Scratch-Lauf sichtbar, bricht aber nicht ab. So bleibt eine bewusste Entscheidung
zwischen mehr Daten, mehr Epochen und einem kleineren Modell möglich.

### Datenqualität und Deduplizierung

Der Audit meldet normalisierte exakte Duplikate, mögliche Near-Duplikate,
potenziell widersprüchliche Antworten, kaputte Ersatzzeichen, Steuerzeichen,
HTML-Boilerplate und stark wiederholte Zeilen. Sichere exakte Duplikate können
automatisch entfernt werden. Near-Duplikate und Qualitätsauffälligkeiten werden
standardmäßig nur gemeldet; ein Ausschluss muss bewusst über `exclude` aktiviert
werden. Beibehaltene Near-Duplikate erben dennoch den Split ihres ersten
Repräsentanten, damit ähnliche Varianten nicht über Training und Validation
verteilt werden.

Nach der Tokenisierung enthält `_producer_meta.json` unter anderem Token- und
Samplezahlen pro Typ, Skip-Gründe, erzeugte Chunks, Packing-Statistiken und
ungefähre Längenperzentile. Validation Loss und Perplexity werden zusätzlich
getrennt für `text` und `dialog` ausgewiesen. Die Live-Diagramme für Loss und
Lernrate stehen nebeneinander und verwenden denselben Optimizer-Step als
Zeitachse; beide Werte stammen immer aus demselben Status-Snapshot.

---

## Training

### Example: Chat / LoRA

```bash
curl -X POST http://127.0.0.1:8002/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "MTSmash/EvaGPT-German-0.7B",
    "csv_path": "./datasets/dein_dataset.csv",
    "save_dir": "./training_outputs",
    "template_mode": "chat",
    "learning_rate": 0.0002,
    "lr_schedule": "cosine",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "max_seq_length": 1024,
    "max_history_turns": null,
    "sort_by_length": true,
    "dynamic_token_batching": true,
    "max_tokens_per_batch": 0,
    "token_normalized_loss": true,
    "rebuild_dataset_cache": true,
    "device": "auto",
    "train_mode": "lora",
    "lora_r": 8,
    "lora_alpha": 16,
    "precision_mode": "auto",
    "gradient_checkpointing": false,
    "merge_lora_on_save": true
  }'
```


### Stop training

```bash
curl -X POST http://127.0.0.1:8002/stop
```

### Training fortsetzen

Jede abgeschlossene Epoche erzeugt standardmäßig einen vollständig fortsetzbaren
Checkpoint unter `checkpoints/checkpoint-XXXXXXXX`. Neben den Modellgewichten werden
Optimizer, Scheduler, AMP-Scaler, Epoche, globaler Schritt und Zufallszustände gesichert.

```json
{
  "resume": "./training_outputs/DEIN_LAUF/checkpoints/checkpoint-00001234",
  "save_every_epoch": true,
  "keep_last_k_checkpoints": 3
}
```

Das Fortsetzen funktioniert für vollständiges Finetuning und für LoRA-Adapter.

### Validation und Early Stopping

Mit `val_split` wird ein stabiler Teil der tokenisierten Samples ausschließlich
für die Validierung reserviert. Chat-Datensätze werden dabei nach vollständigen
Konversationsthreads gruppiert; identische Plain-Text-Samples erhalten ebenfalls
dieselbe Gruppe. So gelangen weder frühere Turns desselben Threads noch identische
Texte in beide Splits. Die Zuordnung basiert auf `split_seed` und bleibt dadurch
auch bei DDP und nach einem Resume identisch.

```json
{
  "val_split": 0.05,
  "split_seed": 42,
  "validate_every_epoch": true,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

Nach jeder Epoche werden die global über alle Ranks gewichtete Validation Loss
und Perplexity berechnet. Das beste Modell liegt anschließend in `best_model/`.
`early_stopping_patience: 0` deaktiviert Early Stopping.

### Profil für 4× NVIDIA V100 32 GB

Bei vier sichtbaren CUDA-GPUs aktiviert der Server DDP automatisch. V100/Volta
verwendet FP16-Tensor-Cores, aber weder natives BF16 noch TF32. Der Worker erkennt
dies automatisch und deaktiviert den wirkungslosen TF32-Pfad. Ein robuster
Ausgangspunkt für LoRA-SFT ist:

```json
{
  "device": "auto",
  "ddp_enabled": true,
  "nproc_per_node": 4,
  "precision_mode": "fp16",
  "max_seq_length": 4096,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "dynamic_token_batching": true,
  "max_tokens_per_batch": 0,
  "max_samples_per_batch": 0,
  "sort_by_length": true,
  "token_normalized_loss": true,
  "dataloader_num_workers": -1,
  "prefetch_factor": 4,
  "gradient_checkpointing": false,
  "cuda_empty_cache_interval_steps": 0,
  "neftune_noise_alpha": 0.0
}
```

`max_tokens_per_batch: 0` berechnet das Budget pro GPU als
`max_seq_length × per_device_train_batch_size`. Der Batchplan gruppiert ähnlich
lange Samples, richtet die Zahl der Batches an vier Ranks und der Gradient
Accumulation aus und zeigt die gemessene Padding-Effizienz im Dashboard. Bei OOM
zuerst das Tokenbudget senken oder Gradient Checkpointing aktivieren. Wenn noch
viel VRAM frei ist, das Tokenbudget schrittweise erhöhen. NEFTune ist bewusst
standardmäßig aus und sollte nur als validiertes Experiment, zum Beispiel mit
`neftune_noise_alpha: 5`, gegen denselben Validation-Split getestet werden.

Für das erste Scratch-Grundtraining auf 4× V100 ist eine kürzere Sequenzlänge
wirtschaftlicher. Ein robuster Startpunkt ist:

```json
{
  "train_from_scratch": true,
  "train_mode": "full",
  "training_phase": "pretrain",
  "precision_mode": "fp16",
  "max_seq_length": 2048,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "warmup_ratio": 0.02,
  "chunk_long_texts": true,
  "text_chunk_overlap": 128,
  "append_eos_to_text": true,
  "deduplicate_exact": true,
  "near_duplicate_action": "warn",
  "quality_filter_mode": "warn"
}
```

Erst nach einem stabilen Grundtraining sollte die Kontextlänge in einem
Folgelauf auf 4096 erhöht werden. Der Worker erweitert bei Scratch-Modellen die
Positionskonfiguration automatisch, falls die Basis-Config kleiner ist.

### Status / Logs / Preview

```bash
curl http://127.0.0.1:8002/status
curl http://127.0.0.1:8002/logs
curl http://127.0.0.1:8002/livepreview
```

WebSocket:

```text
/ws/logs
```

---

## Inference

### Load a model

```bash
curl -X POST http://127.0.0.1:8002/load_inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "./training_outputs/DEIN_MODELL_ORDNER",
    "device": "auto"
  }'
```

### Chat

```bash
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role":"user","content":"Sag Hallo!"}
    ],
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### Streaming Chat

```bash
curl -N -X POST http://127.0.0.1:8002/chat_stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role":"user","content":"Erzähl mir eine kurze Geschichte."}
    ],
    "max_new_tokens": 128,
    "temperature": 0.8,
    "top_p": 0.9
  }'
```

---

## OpenAI-compatible API

Available endpoints:

* `GET /v1/models`
* `POST /v1/chat/completions`
* `POST /v1/completions`

### Auth

Default local API key:

```text
Authorization: Bearer matelix-local-dev-key
```

### List models

```bash
curl http://127.0.0.1:8002/v1/models \
  -H "Authorization: Bearer matelix-local-dev-key"
```

### Chat completions

```bash
curl -X POST http://127.0.0.1:8002/v1/chat/completions \
  -H "Authorization: Bearer matelix-local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./training_outputs/DEIN_MODELL_ORDNER",
    "messages": [
      {"role":"user","content":"Schreibe einen kurzen Reim über KI."}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Streaming SSE

```bash
curl -N -X POST http://127.0.0.1:8002/v1/chat/completions \
  -H "Authorization: Bearer matelix-local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./training_outputs/DEIN_MODELL_ORDNER",
    "messages": [
      {"role":"user","content":"Gib mir 5 Ideen für ein FastAPI Projekt."}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

---

## Important Parameters

| Parameter                | Meaning                                  |
| ------------------------ | ---------------------------------------- |
| `model_dir`              | Hugging Face repo ID or local model path |
| `csv_path`               | path to training CSV                     |
| `template_mode`          | `chat`, `dialogplus`, `plain`            |
| `mixed_training`         | Dialog- und `Text`-Samples gemeinsam trainieren |
| `mixed_text_column`      | Name der zusätzlichen Textspalte; Standard `Text` |
| `training_phase`         | `custom`, `pretrain`, `mixed` oder `sft`        |
| `text_token_weight`      | Zielgewicht freier Texttokens im Mischmodus      |
| `dialog_token_weight`    | Zielgewicht überwachter Dialogtokens              |
| `max_mixture_oversample` | maximale Vervielfachung unterrepräsentierter Daten |
| `max_seq_length`         | maximum token window                     |
| `chunk_long_texts`       | überlange freie Texte automatisch aufteilen       |
| `text_chunk_overlap`     | wiederholte Tokens zwischen benachbarten Chunks   |
| `append_eos_to_text`     | EOS an jedes Text-/Chunk-Ende anhängen            |
| `pack_short_texts`       | kurze Texte optional split-sicher zusammenführen  |
| `deduplicate_exact`      | normalisierte exakte Duplikate vorab entfernen    |
| `near_duplicate_action`  | `off`, `warn` oder `exclude`                      |
| `quality_filter_mode`    | Qualitätsauffälligkeiten melden oder ausschließen |
| `train_scratch_tokenizer` | neuen Tokenizer aus dem aktiven Scratch-Korpus lernen |
| `max_history_turns`      | optional extra turn cap                  |
| `rebuild_dataset_cache`  | rebuild tokenized cache                  |
| `train_mode`             | `full` or `lora`                         |
| `lora_r`                 | LoRA rank                                |
| `lora_alpha`             | LoRA alpha                               |
| `lora_dropout`           | Dropout innerhalb der LoRA-Adapter       |
| `lora_target_modules`    | optionale explizite Modulliste           |
| `resume`                 | Verzeichnis eines Epoch-Checkpoints      |
| `save_every_epoch`       | resumierbaren Checkpoint je Epoche sichern |
| `keep_last_k_checkpoints` | Anzahl aufzubewahrender Checkpoints     |
| `val_split`              | Anteil exklusiver Validation-Samples     |
| `split_seed`             | reproduzierbare Split-Zuordnung           |
| `early_stopping_patience` | Epochen ohne Verbesserung vor Abbruch   |
| `early_stopping_min_delta` | minimale relevante Loss-Verbesserung   |
| `precision_mode`         | `auto`, `fp32`, `fp16`, `bf16`           |
| `gradient_checkpointing` | reduces VRAM usage, slower               |
| `dynamic_token_batching` | Batchgröße an ein Tokenbudget anpassen   |
| `max_tokens_per_batch`   | maximales gepaddetes Tokenbudget pro GPU; `0` = auto |
| `max_samples_per_batch`  | zusätzliche Sample-Obergrenze; `0` = auto |
| `token_normalized_loss`  | Loss über alle Ziel-Tokens und Ranks korrekt gewichten |
| `dataloader_num_workers` | Worker je Rank; `-1` = automatisch       |
| `prefetch_factor`        | vorgeladene Batches je DataLoader-Worker |
| `neftune_noise_alpha`    | optionales NEFTune; `0` = deaktiviert    |
| `dataset_audit_strict`   | bei strukturellen Dataset-Fehlern abbrechen |

---

## DDP / Multi-GPU

Example:

```json
{
  "ddp_enabled": true,
  "nproc_per_node": 4,
  "master_addr": "127.0.0.1",
  "master_port": 29500
}
```

If multiple CUDA GPUs are available, distributed training can be enabled.

---

## LoRA / Merge Behavior

On save:

* adapter is always saved normally
* if `merge_lora_on_save=true`

  * the system also tries to create a merged model
* if `merge_and_unload()` is not supported by the model class

  * training still succeeds
  * adapter remains usable

---

## Troubleshooting

### No shard created / no usable samples

Possible reasons:

* CSV is empty
* wrong column names
* `chunk_long_texts=false` and all text samples are larger than `max_seq_length`
* target answers are too long
* individual turns are too large

### LoRA merge fails

Usually not fatal:

* adapter is still saved
* only the additional merged model is missing

### CUDA is not used

Check:

* correct PyTorch build
* CUDA installation
* `torch.cuda.is_available()`

### UI still shows outdated behavior

Usually solved by:

* hard refresh / clear browser cache
* `rebuild_dataset_cache=true`
* verifying that the correct `static/index.html` is loaded

---

## License

This project is licensed under the **Apache License 2.0**.
See `LICENSE`.

---

## Trademark

**MaTeLiX AI** is a trademark / brand of **TMP-SYSTEM-SERVICE GmbH**.

---

## Recommended Defaults (4× V100 32 GB)

```json
{
  "template_mode": "chat",
  "max_seq_length": 4096,
  "max_history_turns": null,
  "rebuild_dataset_cache": true,
  "train_mode": "lora",
  "lora_r": 8,
  "lora_alpha": 16,
  "precision_mode": "fp16",
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "dynamic_token_batching": true,
  "max_tokens_per_batch": 0,
  "dataloader_num_workers": -1,
  "prefetch_factor": 4,
  "gradient_checkpointing": false,
  "val_split": 0.05,
  "early_stopping_patience": 3
}
```
