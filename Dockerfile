FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# One image for both services: the API and the Streamlit UI.
RUN pip install ".[ui]"

COPY config ./config
COPY data/plans ./data/plans
COPY scripts ./scripts
COPY ui ./ui
COPY .streamlit ./.streamlit
# The database is generated, not committed: same seed, same data.
RUN python scripts/generate_data.py

EXPOSE 8000 8501
CMD ["uvicorn", "benefitpilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
