# AI Agent Coding Directives

When acting on this repository, you must adhere strictly to the following rules, starting with the absolute highest priority constraint:

## 1. CRITICAL: Standard Patch File Edits ONLY (Unified Diff)
- **Format Requirements:** When modifying existing code files, you must output your changes exclusively as a standard unified diff patch (Git-compatible diff format) enclosed in a standard markdown code block (` ```diff `).
- **No Utility Scripts:** Never generate, execute, or rely on temporary Python scripts, shell wrapper commands, or custom file-writing scripts to perform code modifications. 
- **Context & Accuracy:** Ensure file paths, line numbers, and context lines (`@@ ... @@`) are fully accurate so the patch can be applied cleanly via standard tools (`git apply` or `patch`).
- **Minimality:** Keep patches strictly targeted to the requested change. Do not alter surrounding code, formatting, variable names, or whitespace unless required for the fix.

## 2. Code Modification Constraints
* **Minimal Changes:** When working on code output, after reading the input files, make the smallest change possible to achieve the requested result.
* **Preservation:** Do not change any other code, formatting, or names outside the scope of the specific task.
* **Terse Output:** If there are no changes to files, just state so and skip printing a file to the chat to save time and screen space.

## 3. Python Best Practices
* **Type Hinting:** Use explicit Python type hints for all function signatures and complex variables.
* **NumPy Vectorization:** Favor NumPy vectorized operations over Python `for` loops when processing image arrays.
* **Separation of Concerns:** Keep the PyQt6 GUI logic decoupled from the NumPy/OpenCV image processing math.

## 4. Workflow Execution
* Refer to `architecture.md` before implementing any image processing logic.
* Refer to `tasks.md` to understand the current phase of the project. Do not skip ahead or build features that are scheduled for later phases unless explicitly instructed.

## 5. Unit Testing Guidelines
To ensure robust test coverage and catch edge cases, all unit tests must adhere to the following rules:
1. **Non-Square Numpy Arrays:** Any test utilizing numpy arrays must use non-square dimensions (e.g., shape `(13, 12, 3)` or `(4, 2, 3)` instead of `(10, 10, 3)` or `(10, 10)`). This prevents axis-handling errors that remain hidden when row and column dimensions are identical.
2. **Value Flow Verification:** Tests must prove that values actually flow through the logic. Do not solely initialize parameters with `0` or `0.0` and assert `0` or `0.0`. Always include test cases with non-zero, distinct values to positively confirm the code correctly applies transformations, assignments, and state updates.
3. **Single Source of Truth (SSOT) for Diagnostics & Testing:** Never re-implement, duplicate, or independently re-calculate core mathematical transforms, masks (e.g., luminance or shadow masks), or intermediate states inside test scripts or diagnostic plotting functions. Diagnostic and test scripts must consume the exact arrays and values returned directly by the core engine functions to prevent silent drift between test logic and implementation logic.

## 6. Development & Test Environment Setup
Before running any tests, linting, or python scripts, you MUST ensure the Python virtual environment is active and up to date:
1. Check if `ai-venv` exists. If not, create it: `python -m venv ai-venv`, do not use `venv`.
2. Activate it or use its python/pip binaries directly (`ai-venv\Scripts\python` on Windows or `ai-venv/bin/python` in Linux containers).
3. Ensure dependencies are installed: `ai-venv\Scripts\python -m pip install -r requirements.txt` (or inside the container: `pip install -r requirements.txt`).
4. Always run tests using the venv's python or pytest.
5. You need to run pytest in "headless" mode via: `xvfb-run pytest`
6. **Automatic Parameter Integration for Verification:** Once stable, optimal parameters are discovered by the optimization script, they must be automatically integrated and saved into the default values of the core math engine implementation files so that the human can immediately run tests and verify the visual/numerical output without manual parameter copying.

## 7. Color Grading Optimization Protocol
* **RULE 1: DIVIDE AND CONQUER (CHANNEL ISOLATION):** Never optimize all 21 parameters (Red, Green, Blue) simultaneously. The genetic algorithm will stall in the combinatorial expansion. You must isolate optimizations to a single color channel (7 parameters) per run.
* **RULE 2: TWO-STEP MASKING PROCEDURE:** 
  - *Step A (Structure):* Optimize Gaussian structural parameters (widths, gains, centers) by targeting ONLY the grayscale gradient RMSE. Ignore color blocks completely during this step.
  - *Step B (Color Protection):* Lock optimized structural parameters. Optimize Hue Weights targeting RGBCMY color blocks RMSE.
* **RULE 3: COMPUTE BUDGET PRESERVATION:** When searching for slider parameters, do NOT run the general unit test suite. Execute only the specific optimization function requested.
* **RULE 4: SILENT SOLVERS & SPARSE TRACKING:** Always set genetic algorithms to run silently (`disp=False`). Use a tracking list in the objective function to print progress sparsely—only output a line when the error metric drops by **more than 20%**.
* **RULE 5: MANDATORY COMPARATIVE PLOTTING:** Every optimization milestone or final state evaluation MUST automatically execute plotting routines and save visual grid comparisons to disk.