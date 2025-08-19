from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Configurações
QDRANT_URL = os.getenv("https://b6a242d7-b7d3-427d-ac5d-bceafc69d92f.europe-west3-0.gcp.cloud.qdrant.iohttps://b6a242d7-b7d3-427d-ac5d-bceafc69d92f.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RP87OL8e2FhKWEEIKKUV2MdMdsy83tKz_HyiUOYKH-8")
COLLECTION = ("historia")

# Inicialização
app = FastAPI()
client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Cria coleção se não existir
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

# Modelos de dados
class Chunk(BaseModel):
    id: str
    titulo: str
    texto: str
    tags: list[str] = []
    canonical: bool = True

# Endpoints
@app.post("/chunks")
def add_chunk(chunk: Chunk):
    embedding = model.encode(chunk.texto).tolist()
    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=chunk.id,
                vector=embedding,
                payload=chunk.dict()
            )
        ]
    )
    return {"status": "ok", "id": chunk.id}

@app.get("/search")
def search(q: str, k: int = 5):
    vector = model.encode(q).tolist()
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=k
    )
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]



