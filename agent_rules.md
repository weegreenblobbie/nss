# AI Agent Coding Directives

When acting on this repository, you must adhere strictly to the following rules:

## 1. Code Modification constraints
*   **Minimal Changes:** When working on code output, after reading the input files, make the smallest change possible to achieve the requested result.
*   **Preservation:** Do not change any other code, formatting, or names outside the scope of the specific task.
*   **Terse Output:** If there are no changes to files, just state so and skip printing a file to the chat to save time and screen space.

## 2. Python Best Practices
*   **Type Hinting:** Use explicit Python type hints for all function signatures and complex variables.
*   **NumPy Vectorization:** Favor NumPy vectorized operations over Python `for` loops when processing image arrays.
*   **Separation of Concerns:** Keep the PyQt6 GUI logic decoupled from the NumPy/OpenCV image processing math.

## 3. Workflow Execution
*   Refer to `architecture.md` before implementing any image processing logic.
*   Refer to `tasks.md` to understand the current phase of the project. Do not skip ahead or build features that are scheduled for later phases unless explicitly instructed.

## 4. Unit Testing Guidelines
To ensure robust test coverage and catch edge cases, all unit tests must adhere to the following rules:

1. **Non-Square Numpy Arrays:** Any test utilizing numpy arrays must use non-square dimensions (e.g., shape `(13, 12, 3)` or `(4, 2, 3)` instead of `(10, 10, 3)` or `(10, 10)`). This prevents axis-handling errors that remain hidden when row and column dimensions are identical.
2. **Value Flow Verification:** Tests must prove that values actually flow through the logic. Do not solely initialize parameters with `0` or `0.0` and assert `0` or `0.0`. Always include test cases with non-zero, distinct values to positivly confirm the code correctly applies transformations, assignments, and state updates.
3. - **Single Source of Truth (SSOT) for Diagnostics & Testing:** Never re-implement, duplicate, or independently re-calculate core mathematical transforms, masks (e.g., luminance or shadow masks), or intermediate states inside test scripts or diagnostic plotting functions. Diagnostic and test scripts must consume the exact arrays and values returned directly by the core engine functions to prevent silent drift between test logic and implementation logic.

## 5. Development & Test Environment Setup
Before running any tests, linting, or python scripts, you MUST ensure the Python virtual environment is active and up to date:
1. Check if `ai-venv` exists. If not, create it: `python -m venv ai-venv`, do not use `venv`.
2. Activate it or use its python/pip binaries directly (`ai-venv\Scripts\python` on Windows or `ai-venv/bin/python` in Linux containers).
3. Ensure dependencies are installed: `ai-venv\Scripts\python -m pip install -r requirements.txt` (or inside the container: `pip install -r requirements.txt`).
4. Always run tests using the venv's python or pytest.
5. You need to run pytest in "headless" mode via: `xvfb-run pytest`
6. **Automatic Parameter Integration for Verification:** Once stable, optimal parameters are discovered by the optimization script, they must be automatically integrated and saved into the default values of the core math engine implementation files so that the human can immediately run tests and verify the visual/numerical output without manual parameter copying.