# api.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastapi.openapi.docs import get_redoc_html


# -----------------------
# Config via env vars
# No Render, defina:
# QDRANT_URL=https://...qdrant.cloud
# QDRANT_API_KEY=xxxxx
# COLLECTION=historia   (opcional)
# -----------------------
QDRANT_URL = os.getenv("https://b6a242d7-b7d3-427d-ac5d-bceafc69d92f.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RP87OL8e2FhKWEEIKKUV2MdMdsy83tKz_HyiUOYKH-8")
COLLECTION = os.getenv("COLLECTION", "historia")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2

app = FastAPI(title="Memória Externa — Y/S", docs_url=None, redoc_url=None)

@app.get("/docs")
def custom_docs():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title="Memória Externa — ReDoc",
    )


# CORS (ajuste allow_origins para seu domínio quando publicar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
_qdrant = None
_model = None

def get_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise RuntimeError("QDRANT_URL/QDRANT_API_KEY não configurados nas env vars.")
        _qdrant = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)

        # cria coleção se não existir (sem apagar dados)
        try:
            _qdrant.get_collection(COLLECTION)
        except Exception:
            _qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
    return _qdrant

def get_model():
    global _model
    if _model is None:
        # Carrega sob demanda — evita travar o boot do Render
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

class Chunk(BaseModel):
    id: str = Field(..., description="Identificador único do bloco canônico")
    titulo: str
    texto: str
    tags: List[str] = []
    canonical: bool = True

@app.get("/health")
def health():
    # Não toca no modelo nem no Qdrant — responde rápido
    return {"ok": True, "collection": COLLECTION}

@app.post("/chunks")
def add_chunk(chunk: Chunk):
    try:
        model = get_model()
        client = get_client()
        embedding = model.encode(chunk.texto).tolist()
        client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload=chunk.model_dump()
                )
            ]
        )
        return {"status": "ok", "id": chunk.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao inserir chunk: {e}")

@app.post("/chunks/batch")
def add_chunks_batch(chunks: List[Chunk]):
    try:
        model = get_model()
        client = get_client()
        vectors = [model.encode(c.texto).tolist() for c in chunks]
        points = [
            models.PointStruct(
                id=c.id,
                vector=vectors[i],
                payload=chunks[i].model_dump()
            )
            for i, c in enumerate(chunks)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        return {"status": "ok", "count": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no batch: {e}")

@app.get("/search")
def search(q: str, k: int = 5):
    try:
        model = get_model()
        client = get_client()
        vector = model.encode(q).tolist()
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=max(1, min(k, 50))  # sanity cap
        )
        return [
            {
                "id": h.id,
                "score": h.score,
                "payload": h.payload
            }
            for h in hits
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na busca: {e}")

