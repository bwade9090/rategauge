# RateGauge service image: read endpoints serve the committed evaluation
# artifacts baked into the image; POST /extract needs provider keys passed at
# runtime (e.g. --env-file .env — never baked in).
FROM python:3.13-slim

WORKDIR /app

# Layer-cache the dependency install: metadata first, source after.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# The service's data surface: catalogs + golden set + evaluation artifacts
# (documents themselves are fetched on demand and never shipped).
COPY configs ./configs
COPY data/catalog ./data/catalog
COPY data/golden ./data/golden
COPY eval ./eval

EXPOSE 8000
CMD ["rategauge", "serve", "--host", "0.0.0.0", "--port", "8000"]
