ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=1800 \
    UV_HTTP_RETRIES=10 \
    UV_CONCURRENT_DOWNLOADS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace:/opt/repo

ARG INSTALL_UV=1
ARG UV_VERSION=0.13.6
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG UV_INDEX_URL=https://pypi.org/simple
ENV UV_INDEX_URL=${UV_INDEX_URL}
RUN if [ "$INSTALL_UV" = "1" ]; then \
        python -m pip install --no-cache-dir --progress-bar off \
            --timeout 1800 --retries 20 --index-url "$PIP_INDEX_URL" \
            "uv==${UV_VERSION}"; \
    fi
RUN python -c "import setuptools, wheel" 2>/dev/null || \
    python -m pip install --no-cache-dir --progress-bar off \
        --timeout 1800 --retries 20 --index-url "$PIP_INDEX_URL" \
        "setuptools>=61.0" wheel

WORKDIR /opt/repo
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY reasoning_from_scratch ./reasoning_from_scratch
RUN python -m venv --system-site-packages "$UV_PROJECT_ENVIRONMENT"
# Keep the default image focused on the core chapters and tests. Optional
# Appendix F/G dependencies can be added at build time.
ARG INSTALL_EXTRAS=0
ARG USE_PYPI_TORCH=0
RUN if [ "$USE_PYPI_TORCH" = "1" ]; then \
        if [ "$INSTALL_EXTRAS" = "1" ]; then \
            uv sync --no-sources --no-build-isolation --group dev --extra extra; \
        else \
            uv sync --no-sources --no-build-isolation --group dev; \
        fi; \
    elif [ "$INSTALL_EXTRAS" = "1" ]; then \
        uv sync --frozen --no-build-isolation --group dev --extra extra; \
    else \
        uv sync --frozen --no-build-isolation --group dev; \
    fi

COPY . /opt/repo
COPY docker/entrypoint.sh /usr/local/bin/reasoning-entrypoint
RUN chmod +x /usr/local/bin/reasoning-entrypoint

WORKDIR /workspace
ENTRYPOINT ["reasoning-entrypoint"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=reasoning"]
