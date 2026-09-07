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
