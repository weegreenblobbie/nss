FROM python:3.11-slim

# Set up a non-root user
RUN useradd -m -s /bin/bash ubuntu
WORKDIR /home/ubuntu

# Update and install system dependencies required for PyQt6 offscreen rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    bash \
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

# Install core test dependencies (using headless OpenCV!)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir pytest numpy opencv-python-headless

# Node 22 (includes npm), then Claude Code, both as root
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && npm install -g @anthropic-ai/claude-code \
    && claude --version

USER ubuntu
WORKDIR /workspace

# Explicitly add the installation directory to your system PATH
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

ENV CLAUDE_CONFIG_DIR="/home/ubuntu/.claude"

# Verify installation when the container is run
CMD ["claude", "--dangerously-skip-permissions"]
