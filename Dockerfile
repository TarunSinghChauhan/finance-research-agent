FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    sqlmodel \
    asyncpg \
    redis \
    openai \
    httpx \
    tenacity \
    structlog \
    python-dotenv \
    pandas \
    numpy \
    pytest \
    pytest-asyncio \
    ruff

COPY . .

RUN mkdir -p results

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
