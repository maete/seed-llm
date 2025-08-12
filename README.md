# SEED LLM Query Service

FastAPI service integrating with the SEED Platform and a self-hosted LLM (via Ollama) to answer natural language queries on building property data using a local RAG pipeline.

## Key Files

```
app/
├── main.py      # FastAPI entrypoint, middleware, CORS
├── router.py    # API routes, SEED API integration, LLM querying
├── llm_utils.py # Column mapping, category selection, RAG pipeline
├── auth.py      # Optional internal ACCESS_TOKEN verification
requirements.txt # Python dependencies
Dockerfile       # llm_api build instructions
.env             # Environment variables
```

## Requirements

- Python 3.9+
- Docker + NVIDIA Container Toolkit
- SEED Platform API access
- Ollama (Llama 3 model pulled)
- CUDA-enabled GPU (for embeddings on GPU)

## Environment Variables (`.env`)

```
DEBUG=true
SEED_API=http://localhost:80
ACCESS_TOKEN=your_internal_token
```

## Run with Docker Compose

docker-compose.yml

Run:
```bash
docker compose up --build
```

## Run Locally (Without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Main Endpoint

POST `/api/query_llm` – Query LLM with SEED property data.

Requires:
- SEED API Token in `Authorization` header
- JSON body with `prompt`, `cycle_id`, optional `property_ids` and `columns`
