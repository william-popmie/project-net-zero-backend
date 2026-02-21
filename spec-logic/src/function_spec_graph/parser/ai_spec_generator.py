from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anthropic


def generate_pytest_for_function(
    function_qualified_name: str,
    function_source: str,
    file_path: str,
) -> str:
    """
    Use Claude to generate pytest code for an untested function.
    
    Args:
        function_qualified_name: e.g., "src.app.math_utils.add"
        function_source: The actual function source code
        file_path: Original file path for context
    
    Returns:
        Generated pytest code as string
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Cannot generate specs without API access."
        )

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Generate a complete, production-ready pytest test for this Python function.

Function: {function_qualified_name}
Source file: {file_path}

```python
{function_source}
```

Requirements:
- Use pytest conventions (test_function_name)
- Import the function correctly based on the file path
- Cover main use cases and edge cases
- Use clear, descriptive test names
- Add docstrings explaining what each test verifies
- Use assert statements, not unittest style
- Make tests independent and deterministic

Respond with ONLY the Python test code (no markdown, no explanations).
Include all necessary imports at the top."""

    response = client.messages.create(
        model="claude-opus-4-1",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    test_code = response.content[0].text.strip()
    
    # Remove markdown code fencing if present
    if test_code.startswith("```"):
        lines = test_code.split("\n")
        # Find the first ```python or ``` line
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("```"):
                start_idx = i + 1
                break
        
        # Find the closing ``` line
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith("```"):
                end_idx = i
                break
        
        test_code = "\n".join(lines[start_idx:end_idx]).strip()
    
    return test_code


def determine_test_file_path(project_root: Path, source_file_path: str) -> Path:
    """
    Determine where the test file should be placed.
    
    Convention: src/app/math_utils.py -> tests/app/test_math_utils.py
    """
    source_path = Path(source_file_path)
    
    # Remove common source prefixes
    parts = list(source_path.parts)
    if parts and parts[0] in ("src", "lib", "app"):
        parts = parts[1:]
    
    # Build test path
    filename = source_path.stem
    test_filename = f"test_{filename}.py"
    
    test_dir = project_root / "tests"
    # Add subdirectories (e.g., app/) but not the file itself
    if len(parts) > 1:
        test_dir = test_dir / Path(*parts[:-1])
    
    return test_dir / test_filename


def generate_specs_for_untested(
    graph: dict[str, Any],
    project_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Generate pytest specs for all untested functions.
    
    Args:
        graph: The coverage graph output from build_graph
        project_root: Root of the input project
        dry_run: If True, only show what would be generated
    
    Returns:
        Dict with generation results
    """
    from .graph_parser import extract_function_source
    
    untested = graph["coverage"]["untested_list"]
    results = {
        "generated_count": 0,
        "failed_count": 0,
        "dry_run": dry_run,
        "generated_files": [],
        "errors": [],
    }
    
    if not untested:
        print("[OK] All functions already have tests!")
        return results
    
    print(f"Found {len(untested)} untested functions. Generating specs...\n")
    
    # Group by file
    by_file: dict[str, list[dict]] = {}
    for func_info in untested:
        file_path = func_info["file_path"]
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(func_info)
    
    for source_file, functions in by_file.items():
        try:
            test_file_path = determine_test_file_path(project_root, source_file)
            
            print(f"[*] Generating tests for {source_file}")
            print(f"    -> {test_file_path.relative_to(project_root)}")
            
            test_code_parts = []
            
            # Add header
            test_code_parts.append('"""Auto-generated tests by AI Spec Generator."""\n')
            
            for func_info in functions:
                print(f"    - {func_info['name']}...", end=" ")
                
                try:
                    source_code = extract_function_source(
                        project_root / func_info["file_path"],
                        func_info["qualified_name"]
                    )
                    
                    if not source_code:
                        print("[!] Could not extract source")
                        results["failed_count"] += 1
                        results["errors"].append(
                            f"Could not extract source for {func_info['qualified_name']}"
                        )
                        continue
                    
                    if dry_run:
                        # In dry-run, just count what would be generated
                        results["generated_count"] += 1
                        print("[DRY-RUN OK]")
                    else:
                        test_code = generate_pytest_for_function(
                            func_info["qualified_name"],
                            source_code,
                            func_info["file_path"],
                        )
                        
                        test_code_parts.append(f"\n{test_code}\n")
                        results["generated_count"] += 1
                        print("[OK]")
                    
                except Exception as e:
                    print(f"[ERROR] {e}")
                    results["failed_count"] += 1
                    results["errors"].append(
                        f"{func_info['qualified_name']}: {str(e)}"
                    )
            
            if test_code_parts and not dry_run:
                test_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create __init__.py to make tests/app a package (avoids pytest module name conflicts)
                init_py = test_file_path.parent / "__init__.py"
                if not init_py.exists():
                    init_py.write_text('', encoding="utf-8")
                
                # Merge with existing file if present
                if test_file_path.exists():
                    existing = test_file_path.read_text(encoding="utf-8")
                    final_content = existing + "\n\n# Auto-generated additions\n" + "".join(test_code_parts)
                else:
                    final_content = "".join(test_code_parts)
                
                test_file_path.write_text(final_content, encoding="utf-8")
                results["generated_files"].append(str(test_file_path))
            
            print()
            
        except Exception as e:
            print(f"[ERROR] Failed to process {source_file}: {e}\n")
            results["errors"].append(f"{source_file}: {str(e)}")
    
    return results
