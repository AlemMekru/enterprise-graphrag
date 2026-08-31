"""Neo4j persistence and vector search for document chunks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from neo4j import Driver, Transaction
from neo4j.exceptions import Neo4jError

from app.graph.exceptions import VectorIndexConfigurationError, VectorStoreError
from app.graph.schema import CHUNK_CONSTRAINT_CYPHER, DOCUMENT_CONSTRAINT_CYPHER
from app.models.document import Document
from app.models.vector import EmbeddedChunk, VectorRetrievalResult

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VectorIndexConfig:
    """Centralized Neo4j chunk vector-index configuration."""

    name: str
    dimension: int
    similarity_function: Literal["cosine", "euclidean"] = "cosine"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise VectorIndexConfigurationError("Invalid Neo4j vector index name")
        if not 1 <= self.dimension <= 4_096:
            raise VectorIndexConfigurationError(
                "Vector index dimension must be between 1 and 4096"
            )


class Neo4jVectorStore:
    """Persist embedded chunks and run Neo4j vector queries."""

    def __init__(
        self,
        driver: Driver,
        database: str,
        index_config: VectorIndexConfig,
    ) -> None:
        self.driver = driver
        self.database = database
        self.index_config = index_config

    def ensure_schema(self) -> None:
        """Create uniqueness constraints and a compatible vector index."""
        try:
            with self.driver.session(database=self.database) as session:
                session.run(DOCUMENT_CONSTRAINT_CYPHER).consume()
                session.run(CHUNK_CONSTRAINT_CYPHER).consume()
                self._ensure_vector_index(session)
        except Neo4jError as exc:
            raise VectorStoreError("Unable to initialize Neo4j vector schema") from exc

    def upsert_document_chunks(
        self,
        document: Document,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        """Idempotently persist a document, chunks, and HAS_CHUNK edges."""
        invalid_ids = [
            chunk.chunk_id for chunk in chunks if chunk.document_id != document.document_id
        ]
        if invalid_ids:
            raise VectorStoreError(
                "All chunks must reference the document being persisted: "
                f"{', '.join(invalid_ids)}"
            )

        parameters = {
            "document": {
                "document_id": document.document_id,
                "source": document.source,
                "metadata_json": self._dump_metadata(document.metadata),
            },
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "embedding": chunk.embedding,
                    "source_metadata_json": self._dump_metadata(
                        chunk.source_metadata
                    ),
                }
                for chunk in chunks
            ],
        }

        try:
            with self.driver.session(database=self.database) as session:
                session.execute_write(self._upsert_transaction, parameters)
        except Neo4jError as exc:
            raise VectorStoreError("Unable to persist document chunks in Neo4j") from exc

        logger.info(
            "Persisted document chunks",
            extra={"document_id": document.document_id, "chunk_count": len(chunks)},
        )

    def vector_search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorRetrievalResult]:
        """Return the highest-scoring chunks for a query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if len(query_embedding) != self.index_config.dimension:
            raise VectorIndexConfigurationError(
                f"Query embedding has {len(query_embedding)} dimensions; "
                f"index expects {self.index_config.dimension}"
            )

        query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node AS chunk, score
        MATCH (document:Document)-[:HAS_CHUNK]->(chunk)
        RETURN chunk.chunk_id AS chunk_id,
               document.document_id AS document_id,
               chunk.text AS text,
               score,
               chunk.source_metadata_json AS source_metadata_json
        ORDER BY score DESC
        """
        try:
            with self.driver.session(database=self.database) as session:
                records = session.run(
                    query,
                    index_name=self.index_config.name,
                    top_k=top_k,
                    embedding=list(query_embedding),
                )
                return [self._record_to_result(record) for record in records]
        except Neo4jError as exc:
            raise VectorStoreError("Neo4j vector search failed") from exc

    def _ensure_vector_index(self, session: Any) -> None:
        record = session.run(
            """
            SHOW VECTOR INDEXES YIELD name, options
            WHERE name = $index_name
            RETURN options
            """,
            index_name=self.index_config.name,
        ).single()
        if record is not None:
            self._validate_existing_index(record["options"])
            return

        create_query = (
            f"CREATE VECTOR INDEX {self.index_config.name} IF NOT EXISTS "
            "FOR (chunk:Chunk) ON (chunk.embedding) "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {self.index_config.dimension}, "
            "`vector.similarity_function`: "
            f"'{self.index_config.similarity_function}'"
            "}}"
        )
        session.run(create_query).consume()
        logger.info(
            "Created Neo4j vector index",
            extra={"index_name": self.index_config.name},
        )

    def _validate_existing_index(self, options: dict[str, Any]) -> None:
        if not isinstance(options, dict):
            raise VectorIndexConfigurationError(
                f"Vector index '{self.index_config.name}' has invalid options"
            )
        index_config = options.get("indexConfig", {})
        dimension = index_config.get("vector.dimensions")
        similarity = index_config.get("vector.similarity_function")
        normalized_similarity = (
            similarity.lower() if isinstance(similarity, str) else similarity
        )
        if (
            dimension != self.index_config.dimension
            or normalized_similarity != self.index_config.similarity_function
        ):
            raise VectorIndexConfigurationError(
                f"Vector index '{self.index_config.name}' uses dimension={dimension} "
                f"and similarity={similarity}; configured values are "
                f"dimension={self.index_config.dimension} and "
                f"similarity={self.index_config.similarity_function}"
            )

    @staticmethod
    def _upsert_transaction(
        transaction: Transaction, parameters: dict[str, Any]
    ) -> None:
        transaction.run(
            """
            MERGE (document:Document {document_id: $document.document_id})
            SET document.source = $document.source,
                document.metadata_json = $document.metadata_json
            WITH document
            UNWIND $chunks AS chunk_data
            MERGE (chunk:Chunk {chunk_id: chunk_data.chunk_id})
            SET chunk.document_id = chunk_data.document_id,
                chunk.text = chunk_data.text,
                chunk.chunk_index = chunk_data.chunk_index,
                chunk.embedding = chunk_data.embedding,
                chunk.source_metadata_json = chunk_data.source_metadata_json
            MERGE (document)-[:HAS_CHUNK]->(chunk)
            """,
            **parameters,
        ).consume()

    @staticmethod
    def _dump_metadata(metadata: dict[str, Any]) -> str:
        try:
            return json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise VectorStoreError("Metadata must be JSON serializable") from exc

    @staticmethod
    def _record_to_result(record: Any) -> VectorRetrievalResult:
        raw_metadata = record["source_metadata_json"] or "{}"
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError) as exc:
            raise VectorStoreError("Stored chunk metadata is not valid JSON") from exc
        return VectorRetrievalResult(
            chunk_id=record["chunk_id"],
            document_id=record["document_id"],
            text=record["text"],
            score=record["score"],
            source_metadata=metadata,
        )
