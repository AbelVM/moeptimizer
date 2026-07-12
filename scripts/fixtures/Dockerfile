# Minimal image for the users-analytics service.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY users ./users
COPY users.jsonl ./users.jsonl

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "users.cli"]
CMD ["--input", "users.jsonl"]
