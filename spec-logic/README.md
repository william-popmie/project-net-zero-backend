# Spec Logic Parser

Deze map bevat de parser/analyzer-logic met support voor heuristic matching, Claude AI matching, en een volledige **LangGraph workflow** voor geautomatiseerde spec generation.

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

## LangGraph Workflow (Aanbevolen voor volledig proces)

De **LangGraph workflow** is een complete pipeline die automatisch:

1. **AST parsing** - Analyseert je Python project en bouwt de function graph
2. **Test matching** - Matcht test functies met project functies (heuristic of AI)
3. **Test execution & Coverage** - Runt pytest met coverage.py validatie
4. **Spec generation** - Genereert automatisch extra tests als coverage te laag is
5. **Output generation** - Produceert complete AST tree structure met alle data

### Run de workflow

```bash
# PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Run de workflow
python -m function_spec_graph.workflow_cli ../input/sample_project \
  --coverage-threshold 80 \
  --max-iterations 3 \
  --use-ai-matching
```

**Parameters:**

- `--coverage-threshold`: Minimum code coverage % vereist (default: 80)
- `--max-iterations`: Max aantal iteraties voor spec generation (default: 3)
- `--use-ai-matching`: Gebruik Claude AI voor test matching

**Workflow logica:**

- Als tests **falen** OF coverage **< threshold** → genereer extra specs met AI
- Herhaal tot coverage threshold bereikt OF max iterations bereikt
- Output: Complete AST tree structure in `output/langgraph_workflow/`

**Output files:**

- `final_graph.json` - Complete graph met alle nodes en edges
- `final_graph.mmd` - Mermaid diagram
- `final_coverage_report.html` - HTML coverage report
- `ast_summary.txt` - Readable AST tree structure met spec mappings
- `coverage.json` - Pytest coverage data (in project root)

## Run op inputproject (Individuele stappen)

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
