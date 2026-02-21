# LangGraph Workflow Demo

Dit document legt uit hoe de LangGraph workflow werkt en hoe je deze kunt gebruiken.

## Architectuur

De workflow bestaat uit de volgende nodes:

```
┌─────────────┐
│ parse_ast   │ → Analyseert Python project met AST
└──────┬──────┘
       ↓
┌─────────────┐
│ match_tests │ → Matcht test functies met project functies
└──────┬──────┘
       ↓
┌─────────────────────┐
│ execute_tests       │ → Runt pytest + coverage.py
│                     │   Checkt coverage threshold
└──────┬──────────────┘
       ↓
    ┌──────┐
    │ Decision │ → Coverage OK? Tests pass?
    └──┬───┬───┘
       │   │
       │   └──────────────────┐
       ↓                      ↓
┌─────────────┐      ┌──────────────┐
│ output      │      │ generate_    │
│             │      │ specs        │
│ Genereer    │      │              │
│ final AST   │      │ AI genereert │
│ tree +      │      │ extra tests  │
│ reports     │      └──────┬───────┘
└─────────────┘             │
                            │
                            ↓
                    (Loop terug naar execute_tests)
```

## State Flow

De workflow gebruikt een **WorkflowState** TypedDict die door elke node flows:

```python
{
    # Input
    "project_root": Path,
    "use_ai_matching": bool,
    "coverage_threshold": float,
    "max_spec_generation_iterations": int,

    # Intermediate state
    "graph": dict,  # AST graph met nodes en edges
    "test_results": dict,  # Pytest output
    "coverage_percentage": float,

    # Output
    "final_output_path": Path,
    "workflow_complete": bool,
    "errors": list[str]
}
```

## Gebruik

### Basis gebruik (zonder AI spec generation)

```bash
python -m function_spec_graph.workflow_cli ../input/sample_project \
  --coverage-threshold 50
```

Dit runt de workflow en genereert output als coverage ≥ 50%.

### Met AI spec generation

```bash
# Zet API key
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Run workflow
python -m function_spec_graph.workflow_cli ../input/sample_project \
  --coverage-threshold 80 \
  --max-iterations 3 \
  --use-ai-matching
```

De workflow zal:

1. AST parsen + tests matchen
2. Pytest runnen met coverage
3. Als coverage < 80%: AI genereert extra tests
4. Herhaal step 2-3 (max 3x)
5. Output genereren

### Output

Alle output wordt geplaatst in `output/langgraph_workflow/`:

```
output/langgraph_workflow/
├── final_graph.json           # Complete graph met nodes/edges
├── final_graph.mmd            # Mermaid diagram
├── final_coverage_report.html # HTML coverage report
└── ast_summary.txt            # Leesbare AST tree structure
```

Het `ast_summary.txt` bestand bevat een complete tree van alle functies met hun specs:

```
Function: src.app.math_utils.add
  File: src/app/math_utils.py:1
  Specs (1):
    - tests.test_math_utils.test_add_returns_sum (confidence: direct_call)

Function: src.app.math_utils.subtract
  File: src/app/math_utils.py:9
  Specs: [NONE - UNTESTED]
```

## Decision Logic

Na `execute_tests` node, de workflow besluit:

```python
if coverage_passed and tests_passed:
    → go to "output"
elif iteration < max_iterations:
    → go to "generate_specs"
else:
    → go to "output" (geef op na max iterations)
```

## Coverage.py Integratie

De workflow gebruikt **coverage.py** (niet alleen de graph-based coverage):

- Runt: `pytest --cov=src --cov-report=json --cov-report=term`
- Leest: `coverage.json` voor exacte coverage %
- Genereert: HTML reports met line-by-line coverage

Dit geeft **echte code coverage** (welke lines worden gerund), niet alleen "welke functies hebben tests".

## AI Spec Generation Details

Wanneer coverage te laag is:

1. **Identificeer untested functies** uit graph coverage
2. **Extract source code** met AST
3. **Genereer pytest code** via Claude API
4. **Schrijf test files** naar `tests/` directory
5. **Re-parse project** om nieuwe tests te detecteren
6. **Loop terug** naar test execution

De AI genereert production-ready pytest code met:

- Correcte imports
- Edge cases
- Docstrings
- Assert statements

## Voorbeelden

### Voorbeeld 1: Project met goede coverage

```bash
$ python -m function_spec_graph.workflow_cli ../input/sample_project

STEP 1: AST PARSING
✓ Parsed 4 project functions
✓ Found 2 spec functions

STEP 2: TEST MATCHING
✓ Matched 2 test-function pairs
✓ Coverage: 50.0% (2/4 tested)

STEP 3: TEST EXECUTION & COVERAGE ANALYSIS
✓ Running pytest...
✓ Code coverage: 85.0%
✓ Coverage threshold met: 85.0% >= 80.0%

STEP 5: FINAL OUTPUT GENERATION
✓ JSON output: output/langgraph_workflow/final_graph.json
✓ Mermaid diagram: output/langgraph_workflow/final_graph.mmd
✓ HTML coverage report: output/langgraph_workflow/final_coverage_report.html
✓ AST summary: output/langgraph_workflow/ast_summary.txt

WORKFLOW COMPLETED
Final Coverage: 85.0%
```

### Voorbeeld 2: Project met lage coverage → AI generation

```bash
$ python -m function_spec_graph.workflow_cli ../input/sample_project \
    --coverage-threshold 90

STEP 1: AST PARSING
✓ Parsed 4 project functions

STEP 2: TEST MATCHING
✓ Coverage: 50.0% (2/4 tested)

STEP 3: TEST EXECUTION & COVERAGE ANALYSIS
✗ Coverage below threshold: 50.0% < 90.0%

STEP 4: AI SPEC GENERATION (Iteration 1)
Found 2 untested functions. Generating specs...
[*] Generating tests for src/app/math_utils.py
    - subtract... [OK]
    - divide... [OK]

[*] Re-parsing project with new specs...

STEP 3: TEST EXECUTION & COVERAGE ANALYSIS (again)
✓ Running pytest...
✓ Code coverage: 92.0%
✓ Coverage threshold met!

STEP 5: FINAL OUTPUT GENERATION
...

WORKFLOW COMPLETED
Final Coverage: 92.0%
```

## Tips

1. **Start laag**: Begin met `--coverage-threshold 50` en verhoog later
2. **Gebruik AI matching**: `--use-ai-matching` voor betere test-functie matching
3. **Check AST summary**: `ast_summary.txt` toont de complete tree structure
4. **Iteraties**: `--max-iterations 3` voorkomt oneindige loops
5. **Dry-run eerst**: Test met lage threshold om workflow te valideren

## Technische Details

- **LangGraph**: State machine voor workflow orchestration
- **AST**: Python's Abstract Syntax Tree voor code analysis
- **Coverage.py**: Industry-standard coverage tool
- **Claude API**: Voor intelligente test generation
- **Pytest**: Test runner met coverage plugin
