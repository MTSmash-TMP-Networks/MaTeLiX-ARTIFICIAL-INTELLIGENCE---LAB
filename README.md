# MaTeLiX AI Lab (Trainer & Inference Server)

<img width="1940" height="811" alt="Matelix" src="https://github.com/user-attachments/assets/23873d32-74ef-4640-b4c5-f24fb40b738c" />


Ein lokales **FastAPI**-Lab für **Fine-Tuning** (Full / LoRA) und **Inference** inkl. Web-UI-Fallback, Live-Logs, Training-Status, Live-Preview sowie einer **OpenAI-kompatiblen API** (`/v1/*`).

> Datei: `matelix_llm_lab.py`  
> Standard-Port: `8002`

---

## Features

### Training
- **Full Fine-Tuning** oder **LoRA Fine-Tuning** (`train_mode: full|lora`)
- CSV-Datasets (`./datasets/*.csv`)
- Trainingsoutputs + Logs in `./training_outputs/…`
- **Live-Status** (`/status`), **Live-Logs** (`/logs` + WebSocket `/ws/logs`)
- **Live-Preview** eines Batches (`/livepreview`)
- Stop-Mechanik mit sauberem Thread-Ende (Stop-Signal wird gesetzt, `running` erst am Ende zurückgesetzt)

### Tokenizer / Template
- Auto-Setup für MaTeLiX-Chat-Template inkl. Rollen-Tokens:
  - `<|System|>`, `<|Benutzer|>`, `<|Assistentin|>`
- `pad_token` wird gesetzt (falls fehlt / identisch zu `eos`)
- Robust: **System-Message** wird als `role="system"` in `messages` geprepended (kein `system=` Parameter in `apply_chat_template`)

### N-Gramm Optimierung (optional)
- Extrahiert häufige N-Gramme (auch code-lastig) und fügt sie als Tokens hinzu
- Initialisiert neue Token-Embeddings aus Mittelwert der Quell-Tokens
- Speichert Mapping in `ngram_token_map.json` im Output-Ordner

### Inference
- Laden/Entladen von Modellen (`/load_inference`, `/unload_inference`)
- Chat (`/chat`) und Streaming (`/chat_stream`)
- Optional `model_dir` pro Request

### OpenAI-kompatibel (`/v1/*`)
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- API-Key via `Authorization: Bearer ...`

---

## Projektstruktur (Default)

```txt
.
├─ matelix_lab_server.py
├─ datasets/
│  └─ your_dataset.csv
├─ training_outputs/
│  └─ <model>_YYYY-MM-DD_HH-MM-SS/
│     ├─ training.log
│     ├─ train_config.json
│     ├─ (model files)
│     └─ (tokenizer files)
└─ static/
   └─ index.html
````

Wenn `./static/index.html` fehlt, wird automatisch eine kleine Fallback-Seite erzeugt.

---

## Voraussetzungen

* Python 3.11+ empfohlen
* PyTorch (CUDA optional)
* NVIDIA-Setup optional: `nvidia-smi` wird genutzt, wenn verfügbar

### Wichtige Python-Pakete

* `fastapi`, `uvicorn`, `pydantic`
* `torch`, `transformers`, `tokenizers`
* `psutil`
* optional für LoRA: `peft`

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install fastapi uvicorn pydantic psutil torch transformers tokenizers
pip install peft   # optional (nur wenn du LoRA nutzen willst)
```

> Hinweis: Für CUDA nutze die passende PyTorch-Installation entsprechend deiner CUDA-Version.

---

## Start

```bash
python matelix_lab_server.py
```

Dann im Browser öffnen:

* [http://127.0.0.1:8002/](http://127.0.0.1:8002/)

---

## API Quickstart

### Hardware / Systemstatus

```bash
curl http://127.0.0.1:8002/hardware
curl http://127.0.0.1:8002/sysstatus
```

### Datasets / Modelle

```bash
curl http://127.0.0.1:8002/available_datasets
curl http://127.0.0.1:8002/available_models
curl http://127.0.0.1:8002/trainings
```

---

## Training starten

### Beispiel: Full Fine-Tuning

```bash
curl -X POST http://127.0.0.1:8002/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "path/or/hf-repo-id",
    "csv_path": "datasets/your_dataset.csv",
    "device": "auto",
    "train_mode": "full",
    "learning_rate": 0.0004,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 7,
    "num_train_epochs": 3,
    "chunk_size": 1024,
    "template_mode": "chat",
    "shuffle": false,
    "sort_by_length": true
  }'
```

### Beispiel: LoRA Fine-Tuning

```bash
curl -X POST http://127.0.0.1:8002/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "path/or/hf-repo-id",
    "csv_path": "datasets/your_dataset.csv",
    "device": "auto",
    "train_mode": "lora",
    "lora_r": 8,
    "lora_alpha": 16,
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

---

## Dataset-Formate (CSV)

### 1) `template_mode: "column"`

Nutze eine Textspalte (Default: `text`).

**Beispiel-CSV:**

```csv
text
"Hallo, wie geht es dir?"
"Erkläre mir LoRA in einfachen Worten."
```

Start-Konfig:

```json
{
  "template_mode": "column",
  "column_name": "text"
}
```

### 2) `template_mode: "chat"` (Threaded Dialog)

Erwartet CSV mit Thread-IDs, z.B.:

* `id`
* `parent_id`
* `system`
* `Benutzer`
* `Kontext`
* `Assistentin`

Das Lab rekonstruiert Konversationen über `parent_id` → `id` und rendert daraus Chat-Trainingstexte mit Rollen-Tokens.

### 3) `template_mode: "dialogplus"`

Ähnlich wie `chat`, aber mit konsequenten `</s>`-Trennern pro Turn.

---

## Inference (eigene Endpoints)

### Modell laden

```bash
curl -X POST http://127.0.0.1:8002/load_inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "training_outputs/DEIN_OUTPUT_ORDNER",
    "device": "auto"
  }'
```

### Chat (non-stream)

```bash
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Sag Hallo!"}],
    "max_new_tokens": 128,
    "temperature": 0.7
  }'
```

### Chat (stream)

```bash
curl -N -X POST http://127.0.0.1:8002/chat_stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Erzähl mir eine kurze Geschichte."}],
    "max_new_tokens": 128,
    "temperature": 0.8
  }'
```

---

## OpenAI-kompatible API (`/v1/*`)

### Auth

Setze Header:

* `Authorization: Bearer matelix-local-dev-key`

> Key ist aktuell im Code als `OPENAI_COMPAT_API_KEY = "matelix-local-dev-key"` gesetzt.
> Wenn du ihn leer machst, ist Auth deaktiviert.

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
    "model": "training_outputs/DEIN_OUTPUT_ORDNER",
    "messages": [{"role":"user","content":"Schreibe einen kurzen Reim über KI."}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Streaming (SSE)

```bash
curl -N -X POST http://127.0.0.1:8002/v1/chat/completions \
  -H "Authorization: Bearer matelix-local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "training_outputs/DEIN_OUTPUT_ORDNER",
    "messages": [{"role":"user","content":"Gib mir 5 Ideen für ein FastAPI Projekt."}],
    "stream": true,
    "max_tokens": 200
  }'
```

---

## Sicherheitshinweise

* `HF_TRUST_REMOTE_CODE = False` ist **bewusst** als Sicherheits-Voreinstellung gesetzt.
* Wenn du ein Modell nutzt, das `trust_remote_code=True` benötigt, musst du das gezielt anpassen (auf eigenes Risiko).
* CORS ist offen (`allow_origins=["*"]`) – für echte Deployments bitte einschränken.

---

## Troubleshooting

* **CUDA wird nicht genutzt**: Prüfe `torch.cuda.is_available()` und passende Torch-Version.
* **LoRA klappt nicht**: Stelle sicher, dass `peft` installiert ist und dass die Zielmodule erkannt werden. Im Log steht:

  * `[LoRA] Detected targets: ...`
* **Dataset leer**: Prüfe CSV-Encoding (UTF-8), Spaltennamen und Pfade.
* **Tokenizers parallelism warnings**: ist per `TOKENIZERS_PARALLELISM=false` deaktiviert.

---

## Lizenz

Dieses Projekt ist unter der **Apache License 2.0** lizenziert – siehe `LICENSE`.

---

## Roadmap

* [ ] UI Dashboard (Training, Logs, Preview, Inference)
* [ ] Multi-GPU / Distributed Training
* [ ] Dataset-Validierung + Schema-Checks
* [ ] Checkpoint-Resume
