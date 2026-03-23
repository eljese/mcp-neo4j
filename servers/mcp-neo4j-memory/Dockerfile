FROM python:3.12-slim

WORKDIR /app

# Install build dependencies if needed (for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src

# Install project and dependencies
RUN pip install --no-cache-dir .

# Default to STDIO transport as used in Gemini CLI
ENTRYPOINT ["mcp-neo4j-memory"]
