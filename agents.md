# Repository Instructions for AI Agents

## Development & Test Environment Setup
Before running any tests, linting, or python scripts, you MUST ensure the Python virtual environment is active and up to date:
1. Check if `ai-venv` exists. If not, create it: `python -m venv ai-venv`, do not use `venv`.
2. Activate it or use its python/pip binaries directly (`ai-venv\Scripts\python` on Windows or `ai-venv/bin/python` in Linux containers).
3. Ensure dependencies are installed: `ai-venv\Scripts\python -m pip install -r requirements.txt` (or inside the container: `pip install -r requirements.txt`).
4. Always run tests using the venv's python or pytest.
5. You need to run pytest in "headless" mode via: `xvfb-run pytest`