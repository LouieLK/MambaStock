FROM python:3.12-slim

WORKDIR /api

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


