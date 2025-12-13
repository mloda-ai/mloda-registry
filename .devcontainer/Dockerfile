# Use the same VS Code Dev Container base image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

# Set up a non-root user (matches `remoteUser: vscode` in devcontainer.json)
USER root
WORKDIR /workspace

# Fix workspace ownership for vscode user
RUN chown vscode:vscode /workspace

# Install Docker-in-Docker feature
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
  && install -m 0755 -d /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg | tee /etc/apt/keyrings/docker.asc \
  && chmod a+r /etc/apt/keyrings/docker.asc \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(lsb_release -cs) stable" \
       | tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io \
  && rm -rf /var/lib/apt/lists/*

USER vscode

RUN curl -fsSL https://claude.ai/install.sh | bash

# Copy project files with correct ownership
COPY --chown=vscode:vscode ../ /workspace

# Convenience: start in /workspace on shell open
RUN echo 'cd /workspace' >> /home/vscode/.bashrc

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/home/vscode/.local/bin:$PATH"

# Set tox work directory in user's home to avoid volume mount conflicts
ENV TOX_WORK_DIR=/home/vscode/.tox-cache

# Tell uv to use the tox environment instead of creating .venv
ENV UV_PROJECT_ENVIRONMENT=/home/vscode/.tox-cache/python310

# Install tox globally with tox-uv (uses uv internally for faster installs)
RUN uv tool install tox --with tox-uv

# Create tox cache directory with proper ownership
RUN mkdir -p $TOX_WORK_DIR && chown -R vscode:vscode $TOX_WORK_DIR

# Pre-build the tox environment and install dev deps
RUN tox --workdir $TOX_WORK_DIR -e python310 --notest -vv || true && \
    . $TOX_WORK_DIR/python310/bin/activate && \
    uv pip install -e ".[dev]"

# Auto-activate the tox test environment in interactive shells
RUN echo '[ -f /home/vscode/.tox-cache/python310/bin/activate ] && source /home/vscode/.tox-cache/python310/bin/activate' \
      >> /home/vscode/.bashrc

# Default shell
CMD ["bash", "-c", "cd /workspace && exec bash"]

# Recreate tox env inside container
# tox --workdir /home/vscode/.tox-cache -e python310 --recreate -vv
# sudo chown -R vscode:vscode /workspace
