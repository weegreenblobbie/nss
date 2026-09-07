FROM python:3.11-slim

# Update and install system dependencies required for PyQt6 offscreen rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libxkbcommon0 \
    libdbus-1-3 \
    libfontconfig1 \
    libxrender1 \
    libglib2.0-0 \
    libglib2.0-0t64 \
    libxkbcommon-x11-0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Git and Node.js (for Gemini CLI)
RUN apt-get update && apt-get install -y curl git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Gemini CLI globally
RUN npm install -g @google/gemini-cli

# Install core test dependencies (using headless OpenCV!)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir pytest numpy opencv-python-headless

WORKDIR /workspace

# Keep the container alive
CMD ["/bin/bash"]
