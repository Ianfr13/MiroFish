"""
GraphitiService -- synchronous wrapper around graphiti-core's async Graphiti class.
Provides the same interface MiroFish expects, backed by Neo4j.

Replaces: zep_cloud.client.Zep and backend/app/utils/zep_paging.py
"""

import asyncio
import threading
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from pydantic import Field, create_model

from graphiti_core import Graphiti
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    NodeSearchConfig,
    EdgeSearchMethod,
    NodeSearchMethod,
    EdgeReranker,
    NodeReranker,
)

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_service')


class GraphitiService:
    """
    Synchronous wrapper around async graphiti-core.
    Maintains a singleton Graphiti instance and Neo4j driver for connection reuse.
    Thread-safe: the Neo4j driver pool handles concurrent access.
    """

    # ---- Singleton ----
    _instance: Optional['GraphitiService'] = None
    _graphiti: Optional[Graphiti] = None
    _driver: Any = None
    _initialized: bool = False
    _lock = threading.Lock()

    # ---- Per-service state ----
    _ontology_cache: Dict[str, Tuple[Dict, Dict, Dict]] = {}

    def __init__(self, llm_client=None, embedder=None, cross_encoder=None):
        self._llm_client = llm_client
        self._embedder = embedder
        self._cross_encoder = cross_encoder

    # ========== SINGLETON ==========

    @classmethod
    def get_instance(cls) -> 'GraphitiService':
        """Get or create the singleton GraphitiService instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ========== LIFECYCLE ==========

    def _ensure_initialized(self):
        """Lazy async init. Called before any API operation."""
        if self._initialized:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._async_init())
        else:
            import concurrent.futures
            f = concurrent.futures.Future()

            async def _wrap_init():
                try:
                    await self._async_init()
                    f.set_result(None)
                except Exception as e:
                    f.set_exception(e)

            loop.create_task(_wrap_init())
            f.result(timeout=60)

    async def _async_init(self):
        """Initialize the graphiti-core Graphiti instance and Neo4j connection."""
        if self._llm_client is None:
            self._llm_client = OpenAIClient(
                config=LLMConfig(
                    api_key=Config.LLM_API_KEY,
                    base_url=Config.LLM_BASE_URL,
                    model=Config.LLM_MODEL_NAME,
                ),
            )
        if self._embedder is None:
            self._embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=Config.LLM_API_KEY,
                    base_url=Config.LLM_BASE_URL,
                ),
            )
        self._graphiti = Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            llm_client=self._llm_client,
            embedder=self._embedder,
            cross_encoder=self._cross_encoder,
            store_raw_episode_content=True,
        )
        await self._graphiti.build_indices_and_constraints(delete_existing=False)
        self._driver = self._graphiti._clients.driver
        self._initialized = True
        logger.info("GraphitiService initialized (Neo4j connected)")

    def _run_async(self, coro):
        """Run an async coroutine synchronously, handling nested event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            # Already in an event loop (unlikely for Flask, but safe)
            import concurrent.futures
            f = concurrent.futures.Future()

            async def _wrap():
                try:
                    f.set_result(await coro)
                except Exception as e:
                    f.set_exception(e)

            loop.create_task(_wrap())
            return f.result(timeout=600)

    def close(self):
        """Close the Graphiti instance and Neo4j driver connection."""
        if self._graphiti:
            self._run_async(self._graphiti.close())
            self._initialized = False
            logger.info("GraphitiService closed")

    # ========== GRAPH OPS ==========

    def create_graph(self, name: str) -> str:
        """
        Generate a group_id. No API call needed.

        Replaces: client.graph.create(graph_id=..., name=..., description=...)
        """
        self._ensure_initialized()
        group_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        logger.info(f"Created group_id: {group_id} (name={name})")
        return group_id

    def delete_graph(self, group_id: str) -> None:
        """
        Delete all nodes and edges in a group partition.

        Replaces: client.graph.delete(graph_id=...)
        """
        self._ensure_initialized()
        self._run_async(self._async_delete_graph(group_id))
        logger.info(f"Deleted graph: {group_id}")

    async def _async_delete_graph(self, group_id):
        """Internal async: delete all entities and nodes for a group_id."""
        from graphiti_core.nodes import Node as GNode
        from graphiti_core.edges import EntityEdge as GEdge

        driver = self._driver

        # Delete edges first (referential integrity)
        try:
            edges = await GEdge.get_by_group_ids(driver, group_ids=[group_id])
            if edges:
                await GEdge.delete_by_uuids(driver, uuids=[e.uuid for e in edges])
        except Exception as e:
            logger.warning(f"Error deleting edges for {group_id}: {e}")

        # Delete nodes (EpisodicEdges cascade with nodes)
        try:
            await GNode.delete_by_group_id(driver, group_id=group_id)
        except Exception as e:
            logger.warning(f"Error deleting nodes for {group_id}: {e}")

    # ========== ONTOLOGY ==========

    @staticmethod
    def _safe_attr_name(name: str) -> str:
        """Convert reserved attribute names to safe alternatives."""
        RESERVED = {'uuid', 'name', 'group_id', 'name_embedding', 'summary', 'created_at'}
        return f"entity_{name}" if name.lower() in RESERVED else name

    def build_ontology_dicts(self, ontology: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Convert MiroFish ontology format to graphiti entity_types/edge_types/edge_type_map.

        Returns:
            (entity_types, edge_types_dict, edge_type_map)
        """
        entity_types = {}
        edge_types_dict = {}
        edge_type_map = {}
        from typing import Optional as Opt

        for ed in ontology.get("entity_types", []):
            name = ed["name"]
            attrs = {
                self._safe_attr_name(a["name"]): (
                    Opt[str],
                    Field(description=a.get("description", ""), default=None),
                )
                for a in ed.get("attributes", [])
            }
            entity_types[name] = create_model(
                name,
                __doc__=ed.get("description", f"A {name} entity."),
                **attrs,
            )

        for ed in ontology.get("edge_types", []):
            name = ed["name"].upper()
            attrs = {
                self._safe_attr_name(a["name"]): (
                    Opt[str],
                    Field(description=a.get("description", ""), default=None),
                )
                for a in ed.get("attributes", [])
            }
            edge_types_dict[name] = create_model(
                name,
                __doc__=ed.get("description", f"A {name} relationship."),
                **attrs,
            )
            for st in ed.get("source_targets", []):
                key = (st.get("source", "Entity"), st.get("target", "Entity"))
                edge_type_map.setdefault(key, []).append(name)

        return entity_types, edge_types_dict, edge_type_map

    def set_ontology(self, group_id: str, ontology: Dict):
        """
        Cache ontology for a group. Used automatically on add_episode calls.

        Replaces: client.graph.set_ontology(graph_ids=[graph_id], entities=..., edges=...)
        """
        self._ontology_cache[group_id] = self.build_ontology_dicts(ontology)
        logger.info(f"Ontology cached for {group_id}: "
                    f"{len(self._ontology_cache[group_id][0])} entity types, "
                    f"{len(self._ontology_cache[group_id][1])} edge types")

    # ========== EPISODE INGESTION ==========

    def add_text_batches(
        self,
        group_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback=None,
    ) -> List[str]:
        """
        Batch-add text chunks. Processing is SYNCHRONOUS in graphiti-core.
        Returns episode UUIDs immediately after extraction completes.

        Replaces: client.graph.add_batch(graph_id, episodes=EpisodeData list)
                  + _wait_for_episodes loop (no longer needed)
        """
        self._ensure_initialized()
        return self._run_async(
            self._async_add_text_batches(group_id, chunks, batch_size, progress_callback)
        )

    async def _async_add_text_batches(self, group_id, chunks, batch_size, progress_cb):
        eps = []
        etypes, edge_types, etmap = self._ontology_cache.get(group_id, ({}, {}, {}))
        total = len(chunks)
        total_batches = (total + batch_size - 1) // batch_size

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            if progress_cb:
                progress_cb(
                    f" Batch {batch_num}/{total_batches}",
                    (i + len(batch)) / total,
                )

            raw = []
            for c in batch:
                eu = str(uuid.uuid4())
                raw.append(
                    RawEpisode(
                        name=f"chunk-{eu[:8]}",
                        uuid=eu,
                        content=c,
                        source_description="Document text chunk",
                        source=EpisodeType.text,
                        reference_time=datetime.now(),
                    )
                )
                eps.append(eu)

            await self._graphiti.add_episode_bulk(
                bulk_episodes=raw,
                group_id=group_id,
                entity_types=etypes if etypes else None,
                edge_types=edge_types if edge_types else None,
                edge_type_map=etmap if etmap else None,
            )

        return eps

    def add_single_episode(
        self,
        group_id: str,
        episode_body: str,
        source_description: str = "Agent activity",
        source_type: str = "text",
    ) -> str:
        """
        Add a single episode (used by simulation memory updater).

        Replaces: client.graph.add(graph_id, type="text", data=text)
        """
        self._ensure_initialized()
        return self._run_async(
            self._async_add_single(group_id, episode_body, source_description, source_type)
        )

    async def _async_add_single(self, gid, body, sdesc, stype):
        et, ed, em = self._ontology_cache.get(gid, ({}, {}, {}))
        src = EpisodeType.text if stype == "text" else EpisodeType.message
        r = await self._graphiti.add_episode(
            name=f"ep-{uuid.uuid4().hex[:8]}",
            episode_body=body,
            source_description=sdesc,
            reference_time=datetime.now(),
            source=src,
            group_id=gid,
            entity_types=et if et else None,
            edge_types=ed if ed else None,
            edge_type_map=em if em else None,
        )
        return r.episode.uuid

    # ========== RETRIEVAL ==========

    def fetch_all_nodes(self, group_id: str) -> List[EntityNode]:
        """
        Fetch all nodes for a group using cursor pagination.

        Replaces: fetch_all_nodes(client, graph_id) from zep_paging.py
        """
        self._ensure_initialized()
        return self._run_async(self._async_fetch_all(EntityNode, group_id))

    def fetch_all_edges(self, group_id: str) -> List[EntityEdge]:
        """
        Fetch all edges for a group using cursor pagination.

        Replaces: fetch_all_edges(client, graph_id) from zep_paging.py
        """
        self._ensure_initialized()
        return self._run_async(self._async_fetch_all(EntityEdge, group_id))

    async def _async_fetch_all(self, model_cls, group_id):
        all_items = []
        cursor = None
        page_size = 100
        while True:
            kwargs = {"group_ids": [group_id], "limit": page_size}
            if cursor:
                kwargs["uuid_cursor"] = cursor
            page = await model_cls.get_by_group_ids(self._driver, **kwargs)
            if not page:
                break
            all_items.extend(page)
            if len(page) < page_size:
                break
            cursor = getattr(page[-1], "uuid", None)
        return all_items

    def get_node_by_uuid(self, node_uuid: str) -> Optional[EntityNode]:
        """
        Get a single node by UUID.

        Replaces: client.graph.node.get(uuid_=node_uuid)
        """
        self._ensure_initialized()
        return self._run_async(EntityNode.get_by_uuid(self._driver, node_uuid))

    def get_edges_for_node(self, node_uuid: str) -> List[EntityEdge]:
        """
        Get all edges connected to a node.

        Replaces: client.graph.node.get_entity_edges(node_uuid=node_uuid)
        """
        self._ensure_initialized()
        return self._run_async(EntityEdge.get_by_node_uuid(self._driver, node_uuid))

    def get_episode(self, episode_uuid: str):
        """
        Get a single episode by UUID.

        Replaces: client.graph.episode.get(uuid_=ep_uuid)
        """
        self._ensure_initialized()
        return self._run_async(EpisodicNode.get_by_uuid(self._driver, episode_uuid))

    # ========== SEARCH ==========

    def search(
        self,
        group_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> Any:
        """
        Search the graph for entities and edges matching the query.

        Replaces: client.graph.search(query=..., graph_id=..., limit=..., scope=..., reranker=...)
        Returns graphiti SearchResults with .nodes, .edges, .episodes, .communities.
        """
        self._ensure_initialized()
        return self._run_async(self._async_search(group_id, query, limit, scope))

    async def _async_search(self, group_id, query, limit, scope):
        cfg = SearchConfig(limit=limit)
        if scope in ("edges", "both"):
            cfg.edge_config = EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.cosine_similarity, EdgeSearchMethod.bm25],
                reranker=EdgeReranker.rrf,
            )
        if scope in ("nodes", "both"):
            cfg.node_config = NodeSearchConfig(
                search_methods=[NodeSearchMethod.cosine_similarity, NodeSearchMethod.bm25],
                reranker=NodeReranker.rrf,
            )
        return await self._graphiti.search_(query=query, group_ids=[group_id], config=cfg)
