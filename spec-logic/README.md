# Spec Logic Parser

Deze map bevat de parser/analyzer-logic.
De `input` map blijft puur voor het inputproject.

## Setup (Git Bash)

Een venv in deze map is **optioneel**.

### Optie A: met venv (aanbevolen als je lokaal runt)

```bash
cd spec-logic
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
```

### Optie B: zonder venv

```bash
cd spec-logic
pip install -e .
```

## Run op inputproject

```bash
cd spec-logic
python -m function_spec_graph.cli ../input/sample_project --output-json ./output/graph.json --output-mermaid ./output/graph.mmd
```
