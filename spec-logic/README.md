# Function Spec Graph

> AI-powered test coverage analyzer and automated test generator using AST analysis, LangGraph workflows, and Claude AI.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Function Spec Graph is a comprehensive Python testing toolkit that analyzes your codebase using Abstract Syntax Tree (AST) parsing to:

- **Map functions to tests** - Automatically discover and link test functions to source functions
- **Measure coverage** - Calculate both function-level and line-level code coverage
- **Generate missing tests** - Use Claude AI to create production-ready pytest tests for untested code
- **Orchestrate workflows** - LangGraph-powered pipeline for iterative test improvement

Perfect for hackathons, production codebases, and maintaining high test quality standards.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [LangGraph Workflow (Recommended)](#langgraph-workflow-recommended)
  - [Standalone Graph Generation](#standalone-graph-generation)
  - [AI Spec Generation](#ai-spec-generation)
- [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)

---

## Features

✨ **Core Capabilities**

- 🔍 **AST-based Analysis** - Parse Python files without executing code
- 🧪 **Smart Test Matching** - Heuristic + AI matching for test-to-function linking
- 📊 **Coverage Analysis** - Integration with `coverage.py` for accurate metrics
- 🤖 **AI Test Generation** - Claude-powered pytest generation for untested functions
- 🔄 **Iterative Workflow** - Automated loop until coverage threshold is met
- 📈 **Multiple Outputs** - JSON, Mermaid diagrams, HTML reports, and AST summaries

🎯 **Matching Strategies**

| Strategy       | Confidence | Description                                                    |
| -------------- | ---------- | -------------------------------------------------------------- |
| Direct Call    | High       | Detects explicit function calls in test code                   |
| Name Heuristic | Medium     | Matches based on naming conventions (e.g., `test_add` → `add`) |
| AI Matching    | Variable   | Claude analyzes semantic relationships                         |

---

## Prerequisites

- **Python 3.11+** (tested on 3.12)
- **pip** package manager
- **Anthropic API Key** (for AI features) - [Get yours here](https://console.anthropic.com/)

### System Requirements

- OS: Windows 10+, macOS 10.15+, or Linux
- Memory: 2GB RAM minimum
- Disk Space: 500MB for dependencies

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/project-net-zero-backend.git
cd project-net-zero-backend/spec-logic
```

### 2. Create Virtual Environment (Recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

This installs the package in editable mode with all dependencies:

- `langgraph>=0.2.0` - Workflow orchestration
- `langchain-core>=0.3.0` - LangChain foundation
- `anthropic>=0.28.0` - Claude API client
- `coverage>=7.0.0` - Code coverage measurement
- `pytest>=8.3.4` - Test framework
- `pytest-cov>=7.0.0` - Pytest coverage plugin

### 4. Verify Installation

```bash
function-spec-workflow --help
```

Expected output:

```
usage: function-spec-workflow [-h] [--coverage-threshold COVERAGE_THRESHOLD]
                              [--max-iterations MAX_ITERATIONS]
                              [--use-ai-matching]
                              project_path
```

---

## Quick Start

### Basic Workflow (No API Key Required)

Analyze a project and generate coverage reports:

```bash
# Navigate to spec-logic directory
cd spec-logic

# Run workflow on sample project
python -m function_spec_graph.workflow_cli ../input/sample_project \
  --coverage-threshold 60 \
  --max-iterations 1
```

**Output:**

```
============================================================
LANGGRAPH WORKFLOW STARTED
============================================================
Project: C:\...\input\sample_project
Coverage Threshold: 60.0%

STEP 1: AST PARSING
[OK] Parsed 4 project functions

STEP 2: TEST MATCHING
[OK] Matched 2 test-function pairs
[OK] Coverage: 50.0% (2/4 tested)

STEP 3: TEST EXECUTION & COVERAGE ANALYSIS
[OK] Code coverage: 60.0%
[OK] Coverage threshold met!

STEP 5: FINAL OUTPUT GENERATION
[OK] AST summary: output\langgraph_workflow\ast_summary.txt

WORKFLOW COMPLETED
Final Coverage: 60.0%
```

### With AI Test Generation

```bash
# Set API key (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-api-key-here"

# Or Bash/Linux
export ANTHROPIC_API_KEY="sk-ant-api-key-here"

# Run with AI-powered features
python -m function_spec_graph.workflow_cli ../input/sample_project \
  --coverage-threshold 90 \
  --max-iterations 3 \
  --use-ai-matching
```

---

## Usage

### LangGraph Workflow (Recommended)

The complete end-to-end pipeline for production use.

#### Command Structure

```bash
function-spec-workflow <project_path> [OPTIONS]
```

#### Options

| Flag                   | Type  | Default  | Description                                 |
| ---------------------- | ----- | -------- | ------------------------------------------- |
| `project_path`         | Path  | Required | Root directory of Python project to analyze |
| `--coverage-threshold` | Float | 80.0     | Minimum coverage % to pass (0-100)          |
| `--max-iterations`     | Int   | 3        | Max spec generation attempts                |
| `--use-ai-matching`    | Flag  | False    | Enable Claude AI for test matching          |

#### Example: Production-Ready Analysis

```bash
python -m function_spec_graph.workflow_cli ./my_project \
  --coverage-threshold 85 \
  --max-iterations 5 \
  --use-ai-matching
```

#### Workflow Steps

```
┌─────────────┐
│ 1. Parse    │ → AST analysis of source files
│    AST      │
└──────┬──────┘
       ↓
┌─────────────┐
│ 2. Match    │ → Link tests to functions
│    Tests    │
└──────┬──────┘
       ↓
┌─────────────┐
│ 3. Execute  │ → Run pytest + coverage.py
│    Tests    │
└──────┬──────┘
       ↓
    Coverage
    Adequate?
    /      \
  YES      NO
   ↓        ↓
Output  ┌──────────┐
        │ 4. Gen   │ → AI creates tests
        │    Specs │
        └────┬─────┘
             ↓
        (Loop back to step 3)
```

---

### Standalone Graph Generation

For one-off analysis without the full workflow.

#### Basic Analysis

```bash
python -m function_spec_graph.cli ../input/sample_project \
  --output-json ./output/graph.json \
  --output-mermaid ./output/graph.mmd \
  --output-html ./output/coverage_report.html
```

#### With AI Matching

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python -m function_spec_graph.cli ../input/sample_project \
  --output-json ./output/graph.json \
  --use-ai-matching
```

---

### AI Spec Generation

Generate tests for untested functions independently.

#### Dry-Run Mode (Preview)

```bash
python -m function_spec_graph.cli ../input/sample_project \
  --generate-missing-specs \
  --dry-run
```

**Output Example:**

```
Found 2 untested functions. Generating specs...

[*] Generating tests for src/app/math_utils.py
    -> tests\app\test_math_utils.py
    - subtract... [DRY-RUN OK]
    - divide... [DRY-RUN OK]

[OK] Generated 2 test functions
```

#### Production Mode

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python -m function_spec_graph.cli ../input/sample_project \
  --generate-missing-specs \
  --output-json ./output/graph.json
```

Generated tests include:

- Proper imports based on project structure
- Edge case coverage (e.g., division by zero)
- Descriptive docstrings
- Multiple assertions per function
- pytest conventions (fixtures, parametrize, etc.)

---

## Running Tests

### Running Tests on Your Project

The workflow automatically runs pytest on your target project:

```bash
# Workflow runs pytest internally
python -m function_spec_graph.workflow_cli ./your_project
```

### Running Tests on the Sample Project

To validate the installation and test the workflow:

```bash
# Navigate to sample project
cd ../input/sample_project

# Run tests manually
python -m pytest -v

# Expected output:
# tests/test_math_utils.py::test_add_returns_sum PASSED         [ 50%]
# tests/test_math_utils.py::test_multiply_returns_product PASSED [100%]
# ============================== 2 passed in 0.08s ==============================
```

### Run with Coverage

```bash
cd ../input/sample_project

python -m pytest --cov=src --cov-report=html --cov-report=term

# Terminal output shows coverage percentage
# Open htmlcov/index.html in browser for detailed report
```

**Expected Coverage Output:**

```
Name                    Stmts   Miss  Cover
-------------------------------------------
src/__init__.py             0      0   100%
src/app/__init__.py         0      0   100%
src/app/math_utils.py      10      4    60%
-------------------------------------------
TOTAL                      10      4    60%
```

### Validate Workflow End-to-End

Test the complete workflow pipeline:

```bash
cd ../spec-logic

# Run workflow with achievable threshold
python -m function_spec_graph.workflow_cli ../input/sample_project \
  --coverage-threshold 60 \
  --max-iterations 1

# ✅ Should complete successfully and show:
# WORKFLOW COMPLETED
# Final Coverage: 60.0%
```

**Verify Output Files:**

```bash
# Check that all output files were created
ls output/langgraph_workflow/

# Expected files:
# final_graph.json
# final_graph.mmd
# final_coverage_report.html
# ast_summary.txt
```

### Testing on Your Own Project

**Project Requirements:**

1. Source code in `src/` directory
2. Tests in `tests/` directory
3. Proper Python package structure with `__init__.py` files
4. Test file imports match project structure

**Example Project Structure:**

```
your_project/
├── src/
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       └── module.py
├── tests/
│   ├── __init__.py  # Optional but recommended
│   └── test_module.py
└── pytest.ini       # Optional
```

**Run Tests:**

```bash
cd your_project

# 1. Verify pytest works
python -m pytest -v

# 2. Check coverage
python -m pytest --cov=src --cov-report=term

# 3. Run the workflow
cd ../spec-logic
python -m function_spec_graph.workflow_cli ../your_project \
  --coverage-threshold 70
```

---

## Configuration

### Project Structure Requirements

Your target project should follow this structure:

```
your_project/
├── src/                 # Source code
│   ├── __init__.py      # Required for imports
│   └── app/
│       ├── __init__.py  # Required for imports
│       └── module.py
├── tests/               # Test files
│   └── test_module.py
├── pytest.ini           # Optional: pytest config
└── .coveragerc          # Optional: coverage config
```

### pytest.ini Example

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
```

### .coveragerc Example

```ini
[run]
source = src
omit =
    */tests/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

### Environment Variables

| Variable            | Required        | Description                               |
| ------------------- | --------------- | ----------------------------------------- |
| `ANTHROPIC_API_KEY` | For AI features | Claude API key from console.anthropic.com |

**Setup:**

**PowerShell (Windows):**

```powershell
# Temporary (current session)
$env:ANTHROPIC_API_KEY = "sk-ant-api-xxxxx"

# Persistent (user profile)
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-api-xxxxx", "User")
```

**Bash/Linux/macOS:**

```bash
# Temporary (current session)
export ANTHROPIC_API_KEY="sk-ant-api-xxxxx"

# Persistent (add to ~/.bashrc or ~/.zshrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-api-xxxxx"' >> ~/.bashrc
source ~/.bashrc
```

---

## Output Structure

### Workflow Output Directory

```
output/langgraph_workflow/
├── final_graph.json              # Complete graph data
├── final_graph.mmd               # Mermaid diagram
├── final_coverage_report.html    # Interactive HTML report
└── ast_summary.txt               # Human-readable AST tree
```

### ast_summary.txt Example

```
============================================================
AST TREE STRUCTURE SUMMARY
============================================================

Function: src.app.math_utils.add
  File: src/app/math_utils.py:1
  Specs (1):
    - tests.test_math_utils.test_add_returns_sum (confidence: direct_call)

Function: src.app.math_utils.subtract
  File: src/app/math_utils.py:9
  Specs: [NONE - UNTESTED]

============================================================
WORKFLOW SUMMARY
============================================================

Coverage: 75.0%
Tests passed: True
Spec generation iterations: 1
```

### final_graph.json Schema

```json
{
  "metadata": {
    "project_root": "/path/to/project",
    "source_function_count": 4,
    "spec_function_count": 3,
    "edge_count": 3
  },
  "nodes": [
    {
      "id": "src.app.module.function_name",
      "kind": "project_function",
      "name": "function_name",
      "qualified_name": "src.app.module.function_name",
      "file_path": "src/app/module.py",
      "line": 10
    }
  ],
  "edges": [
    {
      "source": "src.app.module.function_name",
      "target": "tests.test_module.test_function_name",
      "relation": "validated_by",
      "confidence": "direct_call"
    }
  ],
  "coverage": {
    "total_functions": 4,
    "tested_functions": 3,
    "coverage_percentage": 75.0,
    "untested_list": [...]
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'langgraph'

**Error:**

```
ModuleNotFoundError: No module named 'langgraph'
```

**Solution:**

```bash
pip install langgraph langchain-core coverage pytest pytest-cov
```

#### 2. ANTHROPIC_API_KEY not set

**Error:**

```
[ERROR] ANTHROPIC_API_KEY environment variable not set
```

**Solution:**

```bash
# PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key"

# Bash/Linux
export ANTHROPIC_API_KEY="sk-ant-your-key"
```

**Verify:**

```bash
# PowerShell
echo $env:ANTHROPIC_API_KEY

# Bash/Linux
echo $ANTHROPIC_API_KEY
```

#### 3. Pytest Import Errors

**Error:**

```
ERROR collecting tests/test_module.py
ModuleNotFoundError: No module named 'src'
```

**Solution:** Ensure your test imports match project structure:

```python
# ✅ Correct
from src.app.module import function_name

# ❌ Incorrect
from app.module import function_name
```

**Quick Fix:**

```bash
# Add src to PYTHONPATH (temporary)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use editable install in your project
pip install -e .
```

#### 4. Coverage Threshold Not Met

**Error:**

```
[FAIL] Coverage below threshold: 45.0% < 80.0%
```

**Solutions:**

- **Lower threshold:** `--coverage-threshold 50`
- **Increase iterations:** `--max-iterations 5`
- **Add API key:** Enable automatic test generation
- **Write tests manually:** Add tests to `tests/` directory

#### 5. Pytest Cache Conflicts

**Error:**

```
import file mismatch: imported module 'test_module' has this __file__ attribute
```

**Solution:**

```bash
# Clean all Python cache files
cd your_project
rm -rf .pytest_cache __pycache__ tests/__pycache__ src/__pycache__

# Windows PowerShell
Remove-Item -Recurse -Force .pytest_cache, __pycache__, tests\__pycache__, src\__pycache__
```

#### 6. No Tests Collected

**Error:**

```
collected 0 items
```

**Solution:**

```bash
# Check test file naming (must start with test_)
ls tests/
# Should see: test_*.py files

# Check test function naming (must start with test_)
# ✅ Correct: def test_addition():
# ❌ Wrong:   def addition_test():

# Verify pytest.ini configuration
cat pytest.ini
```

---

## Advanced Usage

### Custom Coverage Thresholds

Set different thresholds for different project phases:

```bash
# Development: Low threshold
python -m function_spec_graph.workflow_cli ./project --coverage-threshold 50

# Pre-production: Medium threshold
python -m function_spec_graph.workflow_cli ./project --coverage-threshold 75

# Production: High threshold
python -m function_spec_graph.workflow_cli ./project --coverage-threshold 90
```

### CI/CD Integration

**GitHub Actions Example:**

```yaml
# .github/workflows/test-coverage.yml
name: Test Coverage Analysis

on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          cd spec-logic
          pip install -e .

      - name: Run workflow
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          cd spec-logic
          python -m function_spec_graph.workflow_cli ../your_project \
            --coverage-threshold 80 \
            --max-iterations 2

      - name: Upload coverage reports
        uses: actions/upload-artifact@v3
        with:
          name: coverage-reports
          path: spec-logic/output/langgraph_workflow/
```

### Batch Analysis

Analyze multiple projects:

```bash
#!/bin/bash
for project in project1 project2 project3; do
  echo "Analyzing $project..."
  python -m function_spec_graph.workflow_cli ../$project \
    --coverage-threshold 80 \
    --output-json ./output/${project}_graph.json
done
```

### Using as a Python Library

```python
from function_spec_graph import run_workflow
from pathlib import Path

# Run workflow programmatically
final_state = run_workflow(
    project_root=Path("../input/sample_project"),
    use_ai_matching=True,
    coverage_threshold=80.0,
    max_iterations=3
)

print(f"Coverage: {final_state['coverage_percentage']}%")
print(f"Output: {final_state['final_output_path']}")
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

```bash
git clone https://github.com/yourusername/project-net-zero-backend.git
cd project-net-zero-backend/spec-logic

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

pip install -e ".[dev]"
```

### Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Format with Black: `black src/`
- Lint with Ruff: `ruff check src/`

### Testing

Before submitting PR:

```bash
# Run all tests
pytest tests/

# Check coverage
pytest --cov=src --cov-report=term-missing

# Validate workflow
python -m function_spec_graph.workflow_cli ../input/sample_project
```

---

## License

MIT License - see LICENSE file for details

---

## Support

- **Documentation**: [LANGGRAPH_WORKFLOW.md](LANGGRAPH_WORKFLOW.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/project-net-zero-backend/issues)
- **API Docs**: [Anthropic Claude](https://docs.anthropic.com/)

---

**Built with ❤️ for the greatest hackathon of Europe**
