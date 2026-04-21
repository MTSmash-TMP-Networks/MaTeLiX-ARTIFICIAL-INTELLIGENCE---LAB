# MaTeLiX ARTIFICIAL INTELLIGENCE - LAB

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Web%20API-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Training-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

<img width="3323" height="1471" alt="Screenshot 2026-04-21 at 12-03-11 MaTeLiX ARTIFICIAL INTELLIGENCE - LAB" src="https://github.com/user-attachments/assets/61530cb9-7cb3-49c2-a02b-0ca4871e3924" />


Local **LLM training and inference lab** with **FastAPI**, **Web UI**, **DDP / Multi-GPU training**, **LoRA**, **live logs**, **live preview** and an **OpenAI-compatible API**.

> Built for local fine-tuning, chat datasets, text datasets and controlled inference workflows.

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
pip install fastapi uvicorn pydantic psutil torch transformers tokenizers
pip install peft
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

## 3. `template_mode="dialogplus"`

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
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3,
    "max_seq_length": 1024,
    "max_history_turns": null,
    "shuffle": false,
    "sort_by_length": true,
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
| `max_seq_length`         | maximum token window                     |
| `max_history_turns`      | optional extra turn cap                  |
| `rebuild_dataset_cache`  | rebuild tokenized cache                  |
| `train_mode`             | `full` or `lora`                         |
| `lora_r`                 | LoRA rank                                |
| `lora_alpha`             | LoRA alpha                               |
| `precision_mode`         | `auto`, `fp32`, `fp16`, `bf16`           |
| `gradient_checkpointing` | reduces VRAM usage, slower               |

---

## DDP / Multi-GPU

Example:

```json
{
  "ddp_enabled": true,
  "nproc_per_node": 2,
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
* all samples are larger than `max_seq_length`
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

## Recommended Defaults

```json
{
  "template_mode": "chat",
  "max_seq_length": 1024,
  "max_history_turns": null,
  "rebuild_dataset_cache": true,
  "train_mode": "lora",
  "lora_r": 8,
  "lora_alpha": 16
}
```
