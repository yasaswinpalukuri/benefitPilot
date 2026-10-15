FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY data/plans ./data/plans
COPY scripts ./scripts
# The database is generated, not committed: same seed, same data.
RUN python scripts/generate_data.py

EXPOSE 8000
CMD ["uvicorn", "benefitpilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
