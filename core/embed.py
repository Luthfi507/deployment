import numpy as np
from PIL import Image
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(
            self,
            collection_name: str = "cve-detection",
            embed_model: str = "google/embeddinggemma-2"
        ):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = collection_name

        self.model = SentenceTransformer(embed_model)

        # vector dimension 
        self.vector_size = self.model.get_embedding_dimension()

        self._init_collection()

    def _init_collection(self) -> np.ndarray:
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.success(f"Qdrant collection '{self.collection_name}' built successfully")

    def image_encode(self, img_path: str) -> np.ndarray:
        image = Image.open(img_path).convert("RGB")
        embeddding = self.model.encode(image)
        return embeddding

    def text_encode(self, query_text: str) -> np.ndarray:
        embedding = self.model.encode(query_text)
        return embedding

    def store_embedding(self, point_id: int, img_path: str, payload: dict = {}):
        vector = self.image_encode(img_path)
        payloads = {"img_path": img_path, **payload}

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payloads
                )
            ]
        )
        logger.success(f"{img_path} with ID {point_id} stored to Qdrant")

    def search(self, query_text: str, limit: int = 5) -> list:
        logger.info(f"Image searching with query: {query_text}")
        query_vector = self.text_encode(query_text)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit
        ).points


        results = []
        for scored_point in search_results:
            results.append({
                "id": scored_point.id,
                "score": scored_point.score,
                **scored_point.payload
            })

        return results