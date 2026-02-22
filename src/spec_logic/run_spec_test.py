#!/usr/bin/env python3
"""
run_spec_test.py - Execute a generated spec against a function implementation.

Creates a throw-away venv, writes the function + spec into a temp project
structure, runs pytest, and streams the results.

Usage:
    python src/spec_logic/run_spec_test.py
"""
import copy
import subprocess
import sys
import textwrap
import venv
from pathlib import Path
from tempfile import TemporaryDirectory

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

FUNCTION_CODE = textwrap.dedent("""\
    import copy
    import numpy as np

    def update_gd_parameters(parameters, back_caches, lr):
        \"\"\"Update network parameters using vanilla gradient descent.

        Args:
            parameters (dict): Current parameters of the network, with keys:
                - 'W{i}' (np.ndarray): weight matrix for layer i.
                - 'b{i}' (np.ndarray): bias vector for layer i.
            back_caches (dict): Gradients from backpropagation, with keys:
                - 'dW{i}' (np.ndarray): gradient of loss w.r.t. W{i}.
                - 'db{i}' (np.ndarray): gradient of loss w.r.t. b{i}.
            lr (float): Learning rate.

        Returns:
            dict: Updated parameters after one gradient descent step. Each
                W{i} and b{i} is adjusted by subtracting lr * gradient.
        \"\"\"
        parameters = copy.deepcopy(parameters)
        layers = len(parameters) // 2
        for l in range(layers):
            parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - lr * back_caches['dW' + str(l + 1)]
            parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - lr * back_caches['db' + str(l + 1)]
        return parameters
""")

SPEC_CODE = textwrap.dedent("""\
    import pytest
    import numpy as np
    from src.optimizers import update_gd_parameters


    def test_update_gd_parameters_single_layer():
        \"\"\"Test gradient descent update for a single layer network.\"\"\"
        parameters = {
            'W1': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'b1': np.array([0.5, 0.5])
        }
        back_caches = {
            'dW1': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'db1': np.array([0.05, 0.05])
        }
        lr = 0.1

        updated = update_gd_parameters(parameters, back_caches, lr)

        expected_W1 = np.array([[0.99, 1.98], [2.97, 3.96]])
        expected_b1 = np.array([0.495, 0.495])

        assert np.allclose(updated['W1'], expected_W1)
        assert np.allclose(updated['b1'], expected_b1)


    def test_update_gd_parameters_multiple_layers():
        \"\"\"Test gradient descent update for a multi-layer network.\"\"\"
        parameters = {
            'W1': np.ones((2, 2)),
            'b1': np.ones(2),
            'W2': np.ones((2, 2)) * 2,
            'b2': np.ones(2) * 2
        }
        back_caches = {
            'dW1': np.ones((2, 2)) * 0.1,
            'db1': np.ones(2) * 0.05,
            'dW2': np.ones((2, 2)) * 0.2,
            'db2': np.ones(2) * 0.1
        }
        lr = 0.5

        updated = update_gd_parameters(parameters, back_caches, lr)

        assert np.allclose(updated['W1'], np.ones((2, 2)) * 0.95)
        assert np.allclose(updated['b1'], np.ones(2) * 0.975)
        assert np.allclose(updated['W2'], np.ones((2, 2)) * 1.9)
        assert np.allclose(updated['b2'], np.ones(2) * 1.95)


    def test_update_gd_parameters_zero_learning_rate():
        \"\"\"Test that zero learning rate returns unchanged parameters.\"\"\"
        parameters = {
            'W1': np.array([[1.0, 2.0]]),
            'b1': np.array([0.5])
        }
        back_caches = {
            'dW1': np.array([[10.0, 20.0]]),
            'db1': np.array([5.0])
        }

        updated = update_gd_parameters(parameters, back_caches, lr=0.0)

        assert np.allclose(updated['W1'], parameters['W1'])
        assert np.allclose(updated['b1'], parameters['b1'])


    def test_update_gd_parameters_does_not_modify_original():
        \"\"\"Test that the original parameters dict is not modified.\"\"\"
        parameters = {
            'W1': np.array([[1.0]]),
            'b1': np.array([0.5])
        }
        original_W1 = parameters['W1'].copy()
        back_caches = {
            'dW1': np.array([[0.1]]),
            'db1': np.array([0.05])
        }

        update_gd_parameters(parameters, back_caches, lr=0.1)

        assert np.allclose(parameters['W1'], original_W1)
""")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_spec(function_code: str, spec_code: str, module_path: str = "src/optimizers") -> int:
    """
    Create a temp venv, write the function + spec into a project layout, run pytest.

    Args:
        function_code: Python source for the implementation module.
        spec_code:     Python source for the pytest spec file.
        module_path:   Dotted import path the spec uses (e.g. 'src.optimizers').
                       Converted to a file path under the temp project root.

    Returns:
        pytest exit code (0 = all passed).
    """
    with TemporaryDirectory(prefix="spec_run_") as tmp:
        tmp = Path(tmp)

        # --- project layout ---------------------------------------------------
        # src/__init__.py  +  src/optimizers.py
        mod_file = tmp / module_path.replace(".", "/")
        mod_file = mod_file.with_suffix(".py")
        mod_file.parent.mkdir(parents=True, exist_ok=True)
        (mod_file.parent.parent / "__init__.py").touch()   # root __init__ if needed
        (mod_file.parent / "__init__.py").touch()
        mod_file.write_text(function_code)

        # tests/test_spec.py
        tests_dir = tmp / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").touch()
        (tests_dir / "test_spec.py").write_text(spec_code)

        # pytest.ini - makes `tmp` the rootdir so `src.*` imports resolve
        (tmp / "pytest.ini").write_text("[pytest]\n")

        # --- venv -------------------------------------------------------------
        venv_dir = tmp / ".venv"
        print(f"[venv] Creating virtual environment at {venv_dir} ...")
        venv.create(str(venv_dir), with_pip=True, clear=True)

        pip = venv_dir / "bin" / "pip"
        python = venv_dir / "bin" / "python"

        print("[venv] Installing pytest and numpy ...")
        subprocess.run(
            [str(pip), "install", "-q", "pytest", "numpy"],
            check=True,
        )

        # --- pytest -----------------------------------------------------------
        print("[pytest] Running tests ...\n")
        result = subprocess.run(
            [str(python), "-m", "pytest", str(tests_dir), "-v"],
            cwd=str(tmp),
        )

        return result.returncode


if __name__ == "__main__":
    exit_code = run_spec(FUNCTION_CODE, SPEC_CODE)
    sys.exit(exit_code)
