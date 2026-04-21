# MaTeLiX ARTIFICIAL INTELLIGENCE - LAB

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Web%20API-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Training-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

Lokales **LLM Training- und Inference-Lab** mit **FastAPI**, **Web-UI**, **DDP-/Multi-GPU-Training**, **LoRA**, **Live-Logs**, **Live-Preview** und einer **OpenAI-kompatiblen API**.

> Entwickelt für lokales Fine-Tuning, Chat-Datasets, Text-Datasets und kontrollierte Inference-Workflows.

---

## Inhaltsverzeichnis

- [Features](#features)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Start](#start)
- [Web-UI](#web-ui)
- [Dataset-Formate](#dataset-formate)
- [Strict Whole-Turn Packing](#strict-whole-turn-packing)
- [Training](#training)
- [Inference](#inference)
- [OpenAI-kompatible API](#openai-kompatible-api)
- [Wichtige Parameter](#wichtige-parameter)
- [DDP / Multi-GPU](#ddp--multi-gpu)
- [LoRA / Merge-Verhalten](#lora--merge-verhalten)
- [Troubleshooting](#troubleshooting)
- [Lizenz](#lizenz)

---

## Features

### Training
- Full Fine-Tuning oder LoRA Fine-Tuning
- Single-GPU, CPU, MPS oder **DDP / Multi-GPU**
- CSV-basierte Trainingsdaten
- Trainingsstatus, Logs und Live-Preview
- Stop-Funktion für laufende Trainings
- automatische Trainingsoutputs pro Run

### Dataset-Verarbeitung
- `chat`, `dialogplus`, `plain`
- Thread-Rekonstruktion über `id` / `parent_id`
- **strict whole-turn packing**
- keine halben Turns / keine halb abgeschnittenen Dialoge
- Oversize-Samples werden übersprungen
- tokenisierte **Shard-Caches** für schnellere Folge-Trainings

### Tokenizer / Template
- automatische Rollen-Tokens:
  - `<|System|>`
  - `<|Benutzer|>`
  - `<|Assistentin|>`
- automatisches `pad_token`-Handling
- eigenes MaTeLiX Chat-Template
- strikte Rollenvalidierung im Chat-Modus

### Inference
- Modell laden / entladen
- normaler Chat
- Streaming-Chat
- LoRA-Adapter oder reguläre Modelle laden
- bevorzugt aktuelles / letztes Modell

### API / UI
- FastAPI-Backend
- OpenAI-kompatible `/v1/*` Endpunkte
- Web-UI mit Hardware-Anzeige, Logs, Loss-Kurve und Chat

---

## Projektstruktur

```text
.
├─ matelix_lab_server_web_ddp.py
├─ matelix_ddp_worker.py
├─ datasets/
│  └─ *.csv
├─ static/
│  └─ index.html
├─ training_outputs/
│  └─ <model>_YYYY-MM-DD_HH-MM-SS/
│     ├─ train_config.json
│     ├─ worker_config.json
│     ├─ training.log
│     ├─ status.json
│     ├─ livepreview.json
│     ├─ dataset_cache/
│     ├─ template_info.json
│     ├─ merged/                # falls LoRA-Merge erfolgreich
│     └─ ...
└─ README.md
````

---

## Installation

### Voraussetzungen

* Python 3.10+ empfohlen
* PyTorch
* `transformers`
* `fastapi`
* `uvicorn`
* `psutil`
* optional: `peft` für LoRA

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Pakete installieren:

```bash
pip install -U pip
pip install fastapi uvicorn pydantic psutil torch transformers tokenizers
pip install peft
```

---

## Start

```bash
python matelix_lab_server_web_ddp.py
```

Danach:

```text
http://127.0.0.1:8002/
```

---

## Web-UI

Die Oberfläche liegt unter:

```text
/static/index.html
```

Funktionen der UI:

* Modellauswahl
* Dataset-Auswahl
* Trainingsparameter
* optionaler History-Deckel
* Cache-Neuaufbau
* LoRA-Optionen
* Live-Logs
* Loss-Kurve
* Live-Preview
* Browser-Chat mit geladenem Modell

---

## Dataset-Formate

## 1. `template_mode="plain"`

Für reine Textdaten.

```csv
text
Das ist ein Beispielsatz.
Noch ein Beispielsatz.
```

Beispiel:

```json
{
  "template_mode": "plain",
  "column_name": "text"
}
```

---

## 2. `template_mode="chat"`

Für Chat-/Thread-Datasets.

Erwartete Felder:

* `id`
* `parent_id`
* `system` (optional)
* `Benutzer`
* `Kontext` (optional)
* `Assistentin`

Beispiel:

```csv
id,parent_id,system,Benutzer,Kontext,Assistentin
1,,Du bist ein freundlicher Chatbot.,Hallo!,,"Hallo! Wie kann ich dir helfen?"
2,1,,Wie spät ist es?,,"Ich habe keinen Zugriff auf deine Uhr, aber du kannst oben rechts schauen."
3,,Du bist Übersetzer.,Übersetze: "Guten Morgen",,Auf Englisch: "Good morning".
```

---

## 3. `template_mode="dialogplus"`

Wie `chat`, aber blockorientiert.

Auch hier gilt:

* ganze Turns
* keine halben Blöcke
* zu große Samples werden übersprungen

---

## Strict Whole-Turn Packing

Die aktuelle Logik ist **tokenbudget-gesteuert**:

* komplette Turns werden rückwärts gesammelt
* nur vollständige Blöcke werden übernommen
* Antwort bleibt vollständig
* kein harter Cut mitten im Turn
* zu große Samples werden verworfen

### Empfehlung

Für normale Chat-Datasets:

```json
{
  "max_history_turns": null
}
```

Das bedeutet:

* nur das Tokenfenster entscheidet
* kein künstlicher Zusatzdeckel

Optional kann `max_history_turns` gesetzt werden, wenn zusätzlich begrenzt werden soll.

---

## Training

### Beispiel: Chat / LoRA

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

### Training stoppen

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

### Modell laden

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

### Streaming-Chat

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

## OpenAI-kompatible API

Verfügbare Endpunkte:

* `GET /v1/models`
* `POST /v1/chat/completions`
* `POST /v1/completions`

### Auth

Standardmäßig:

```text
Authorization: Bearer matelix-local-dev-key
```

### Modelle listen

```bash
curl http://127.0.0.1:8002/v1/models \
  -H "Authorization: Bearer matelix-local-dev-key"
```

### Chat Completions

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

## Wichtige Parameter

| Parameter                | Bedeutung                                    |
| ------------------------ | -------------------------------------------- |
| `model_dir`              | Hugging Face Repo-ID oder lokaler Modellpfad |
| `csv_path`               | Pfad zur Trainings-CSV                       |
| `template_mode`          | `chat`, `dialogplus`, `plain`                |
| `max_seq_length`         | maximales Tokenfenster                       |
| `max_history_turns`      | optionaler zusätzlicher Turn-Deckel          |
| `rebuild_dataset_cache`  | Cache neu erzeugen                           |
| `train_mode`             | `full` oder `lora`                           |
| `lora_r`                 | LoRA Rank                                    |
| `lora_alpha`             | LoRA Alpha                                   |
| `precision_mode`         | `auto`, `fp32`, `fp16`, `bf16`               |
| `gradient_checkpointing` | spart VRAM, kostet Laufzeit                  |

---

## DDP / Multi-GPU

Beispiel:

```json
{
  "ddp_enabled": true,
  "nproc_per_node": 2,
  "master_addr": "127.0.0.1",
  "master_port": 29500
}
```

Wenn mehrere CUDA-Geräte vorhanden sind, kann das Training verteilt gestartet werden.

---

## LoRA / Merge-Verhalten

Beim Speichern gilt:

* Adapter wird regulär gespeichert
* falls `merge_lora_on_save=true`:

  * zusätzlich wird ein Merge versucht
* falls `merge_and_unload()` vom Modell nicht unterstützt wird:

  * Training bleibt trotzdem erfolgreich
  * Adapter bleibt normal nutzbar

---

## Troubleshooting

### Kein erster Shard / kein Cache

Mögliche Gründe:

* CSV ist leer
* Spaltennamen passen nicht
* alle Samples sind größer als `max_seq_length`
* Antwort oder Turns sind einzeln zu groß

### LoRA Merge schlägt fehl

Das ist nicht zwingend kritisch:

* Adapter bleibt gespeichert
* nur das zusätzliche `merged/` Modell fehlt dann

### CUDA wird nicht genutzt

Prüfen:

* passende Torch-Version
* CUDA-Installation
* `torch.cuda.is_available()`

### UI zeigt alte Daten

Meist hilft:

* Browser-Cache leeren
* `rebuild_dataset_cache=true`
* sicherstellen, dass die richtige `static/index.html` geladen wird

---

## Lizenz

Dieses Projekt steht unter der **Apache License 2.0**.
Siehe `LICENSE`.

---

## Trademark

**MaTeLiX AI** ist eine Marke / Brand der **TMP-SYSTEM-SERVICE GmbH**.

---

## Empfohlene Defaults

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
