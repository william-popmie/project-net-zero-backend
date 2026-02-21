# Spec Logic Parser

Deze map bevat de parser/analyzer-logic met support voor heuristic matching en optioneel Claude AI matching.

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

### Heuristic matching (default, snel)

```bash
python -m function_spec_graph.cli ../input/sample_project --output-json ./output/graph.json --output-mermaid ./output/graph.mmd --output-html ./output/coverage_report.html
```

### AI-enhanced matching (Claude API, genauer)

Vereist `ANTHROPIC_API_KEY` environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # Or set in Windows: set ANTHROPIC_API_KEY=...

python -m function_spec_graph.cli ../input/sample_project \
  --output-json ./output/graph.json \
  --output-mermaid ./output/graph.mmd \
  --output-html ./output/coverage_report.html \
  --use-ai-matching
```

### Auto-generate tests voor untested functions (AI Spec Generator)

Gebruik `--generate-missing-specs` om automatisch pytest tests te genereren voor alle functies zonder coverage:

```bash
# Preview wat er gegenereerd zou worden (dry-run)
python -m function_spec_graph.cli ../input/sample_project \
  --generate-missing-specs \
  --dry-run

# Genereer daadwerkelijk test files
python -m function_spec_graph.cli ../input/sample_project \
  --generate-missing-specs \
  --output-json ./output/graph.json
```

Features:

- ✅ Analyseert functie source code met AST
- ✅ Gebruikt Claude om production-ready pytest tests te genereren
- ✅ Respecteert project structure (src/app/math_utils.py → tests/test_math_utils.py)
- ✅ Voegt toe aan bestaande test files of creëert nieuwe
- ✅ Genereert docstrings, edge cases, en clear assertions

**Vereisten**: `ANTHROPIC_API_KEY` environment variable moet gezet zijn.

## Matching Strategies

- **Direct call** (confidence: high): Heuristic finds explicit function calls
- **Name heuristic** (confidence: medium): Test name contains function name
- **AI matching** (confidence: variable): Claude analyzes code semantics

AI fallback only activates if heuristic finds no matches.

## Output

- `graph.json`: Full graph with coverage analysis (reconstructible project data)
- `graph.mmd`: Mermaid diagram for documentation
- `coverage_report.html`: Interactive HTML coverage report
