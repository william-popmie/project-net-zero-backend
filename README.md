# Project Net-Zero — Backend

An AI-powered backend service that automatically optimizes Python AI training code for energy efficiency. Upload a ZIP of your project, get back a leaner, greener version with measured CO₂ savings.

---

## How It Works

The service exposes a single REST endpoint (`POST /optimize`). You upload a ZIP of your Python AI training project, and the pipeline runs three phases:

**Phase 1 — Spec & Test Generation**
Each Python function in the project is parsed and passed through a LangGraph workflow. Claude analyzes the function and generates test specs that capture the function's intended behavior.

**Phase 2 — Optimization & Emissions Measurement**
Each function is optimized by Claude with energy efficiency as the primary objective. Before and after each rewrite, CodeCarbon runs the function in an isolated subprocess and measures real CO₂ emissions (kg). Only improvements that maintain correctness and reduce emissions are kept.

**Phase 3 — In-Place Rewrite**
The optimized function bodies are spliced back into the original source files. The output is re-zipped and returned to the caller.

---

## Flowchart

```
User / Frontend
      │
      │  POST /optimize  (multipart: project.zip)
      ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI server                    │
│                    server.py                        │
└────────────────────┬────────────────────────────────┘
                     │
                     │  extract zip → input-repo/
                     │  discover project root
                     │  pip install requirements.txt
                     ▼
┌─────────────────────────────────────────────────────┐
│              PHASE 1 — Spec Logic                   │
│          src/spec_logic/langgraph_workflow.py        │
│                                                     │
│  ┌─────────────┐     ┌───────────────────────┐      │
│  │  Parser     │────▶│  LangGraph Workflow    │      │
│  │  (AST scan) │     │  Claude: generate      │      │
│  │             │     │  test spec per fn      │      │
│  └─────────────┘     └───────────┬───────────┘      │
│                                  │ spec_logic/       │
│                                  │ output/results.json│
└──────────────────────────────────┼──────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────┐
│              PHASE 2 — Optimizer                    │
│           src/optimizer_logic/optimizer.py          │
│                                                     │
│  for each function:                                 │
│    1. measure baseline CO₂  (CodeCarbon)            │
│    2. Claude rewrites function for efficiency       │
│    3. measure optimized CO₂ (CodeCarbon)            │
│    4. run generated tests → accept if passing       │
│                                                     │
│                      optimizer_logic/output/result.json│
└──────────────────────────────────┬──────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────┐
│              PHASE 3 — Convertor                    │
│          src/convertor/inplace_rewriter.py          │
│                                                     │
│  Splice optimized function bodies back into         │
│  original source files  →  output-repo/             │
└──────────────────────────────────┬──────────────────┘
                                   │
                                   │  stream optimized.zip
                                   ▼
                             User / Frontend
```

---

## Requirements for Submitted Repositories

> **Your project must meet both requirements below, or the optimizer will reject it.**

| Requirement | Details |
|---|---|
| Python AI training program | The repository must contain Python (`.py`) source files that implement a machine-learning or AI training workload (model training loops, data preprocessing, loss computation, etc.) |
| `requirements.txt` present | A `requirements.txt` file must exist at the project root. This is how the optimizer installs your dependencies before analyzing and running your code. |

A minimal valid project looks like:

```
my-ai-project/
├── requirements.txt       ← required
├── train.py               ← your training code
└── model.py
```

---

## Setup

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd project-net-zero-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Start the server

```bash
uvicorn server:app --reload
```

The API is now available at `http://localhost:8000`.

---

## API Reference

### `GET /health`

Returns `{"status": "ok"}` — use this to verify the server is running.

### `POST /optimize`

Accepts a multipart form upload and returns an optimized ZIP archive.

| Field | Type | Description |
|---|---|---|
| `project` | file (`.zip`) | ZIP archive of your Python AI training project |
| `entrypoint` | string (optional) | Hint for the project entrypoint file |

**Example with `curl`:**

```bash
curl -X POST http://localhost:8000/optimize \
  -F "project=@my-ai-project.zip" \
  -o optimized.zip
```

**Response:** `application/zip` — the optimized project ready to unzip and run.

---

## Project Structure

```
project-net-zero-backend/
├── server.py                    # FastAPI app & /optimize endpoint
├── requirements.txt             # Backend dependencies
├── .env.example                 # Environment variable template
├── input-repo/                  # Extracted user project (runtime, auto-managed)
├── output-repo/                 # Optimized output files (runtime, auto-managed)
└── src/
    ├── main.py                  # Pipeline orchestrator (also runnable as CLI)
    ├── parser/
    │   └── graph_parser.py      # AST-based function extractor
    ├── spec_logic/
    │   ├── langgraph_workflow.py # LangGraph orchestration for spec generation
    │   └── ai_spec_generator.py  # Claude prompt for test spec generation
    ├── optimizer_logic/
    │   ├── optimizer.py          # Core optimization loop
    │   ├── emissions.py          # CodeCarbon emissions measurement
    │   └── function_spec.py      # FunctionSpec dataclass
    └── convertor/
        ├── inplace_rewriter.py   # Splices optimized code back into source files
        └── json_to_python.py     # Writes optimized files to output-repo
```

---

## Running the Pipeline via CLI

You can also run the optimizer directly against a local project folder without starting the server:

```bash
python src/main.py /path/to/your/project
```

Output files are written to `src/spec_logic/output/results.json` and `src/optimizer_logic/output/result.json`.
