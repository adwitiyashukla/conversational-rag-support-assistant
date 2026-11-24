FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md config.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV CONVRAG_SERVER_HOST=0.0.0.0

EXPOSE 8000

VOLUME ["/app/data"]

CMD ["convrag", "serve"]
