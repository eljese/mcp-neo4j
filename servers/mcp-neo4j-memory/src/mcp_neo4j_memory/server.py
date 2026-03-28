import json
import logging
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent, ToolAnnotations
from neo4j.exceptions import Neo4jError
from pydantic import Field
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from neo4j import AsyncGraphDatabase

from .neo4j_memory import (
    Entity,
    Feedback,
    Neo4jMemory,
    ObservationAddition,
    ObservationDeletion,
    Relation,
)
from .utils import format_namespace
from .vector_engine import GeminiVectorEngine, OllamaVectorEngine

# Set up logging
logger = logging.getLogger("mcp_neo4j_memory")
logger.setLevel(logging.INFO)


def create_mcp_server(memory: Neo4jMemory, namespace: str = "") -> FastMCP:
    """Create an MCP server instance for memory management."""

    namespace_prefix = format_namespace(namespace)
    mcp: FastMCP = FastMCP("mcp-neo4j-memory")

    @mcp.tool(
        name=namespace_prefix + "read_graph",
        annotations=ToolAnnotations(
            title="Read Graph",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def read_graph() -> ToolResult:
        """Read the entire knowledge graph with all entities and relationships.

        Returns the complete memory graph including all stored entities and their relationships.
        Use this to get a full overview of stored knowledge.

        Returns:
            KnowledgeGraph: Complete graph with all entities and relations

        Example response:
        {
            "entities": [
                {"name": "John Smith", "type": "person", "observations": ["Works at Neo4j"]},
                {"name": "Neo4j Inc", "type": "company", "observations": ["Graph database company"]}
            ],
            "relations": [
                {"source": "John Smith", "target": "Neo4j Inc", "relationType": "WORKS_AT"}
            ]
        }
        """
        logger.info(f"MCP tool: read_graph")
        try:
            result = await memory.read_graph()
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error reading full knowledge graph: {e}")
            raise ToolError(f"Neo4j error reading full knowledge graph: {e}") from e
        except Exception as e:
            logger.error(f"Error reading full knowledge graph: {e}")
            raise ToolError(f"Error reading full knowledge graph: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "create_entities",
        annotations=ToolAnnotations(
            title="Create Entities",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def create_entities(
        entities: list[Entity] = Field(
            ...,
            description="List of entities to create with name, type, observations, and asserted_by",
        ),

    ) -> ToolResult:
        """Create multiple new entities in the knowledge graph.

        Creates new memory entities with their associated observations.
        Note: The graph uses an append-only bitemporal ledger. This sets up the initial facts.

        Returns:
            list[Entity]: The created entities with their final state

        Example call:
        {
            "entities": [
                {
                    "name": "Alice Johnson",
                    "type": "person",
                    "observations": ["Software engineer", "Lives in Seattle"],
                    "asserted_by": "Global Architect"
                }
            ]
        }
        """
        logger.info(
            f"MCP tool: create_entities ({len(entities)} entities, kwargs: {kwargs})"
        )
        try:
            entity_objects = [Entity.model_validate(entity) for entity in entities]
            result = await memory.create_entities(entity_objects)
            return ToolResult(
                content=[
                    TextContent(
                        type="text", text=json.dumps([e.model_dump() for e in result])
                    )
                ],
                structured_content={"result": result},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error creating entities: {e}")
            raise ToolError(f"Neo4j error creating entities: {e}") from e
        except Exception as e:
            logger.error(f"Error creating entities: {e}")
            raise ToolError(f"Error creating entities: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "create_relations",
        annotations=ToolAnnotations(
            title="Create Relations",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def create_relations(
        relations: list[Relation] = Field(
            ..., description="List of relations to create between existing entities"
        ),

    ) -> ToolResult:
        """Create multiple new relationships between existing entities in the knowledge graph.

        Creates directed relationships between entities that already exist. Both source and target
        entities must already be present in the graph. Use descriptive relationship types.

        Returns:
            list[Relation]: The created relationships

        Example call:
        {
            "relations": [
                {
                    "source": "Alice Johnson",
                    "target": "Microsoft",
                    "relationType": "WORKS_AT"
                }
            ]
        }
        """
        logger.info(
            f"MCP tool: create_relations ({len(relations)} relations, kwargs: {kwargs})"
        )
        try:
            relation_objects = [
                Relation.model_validate(relation) for relation in relations
            ]
            result = await memory.create_relations(relation_objects)
            return ToolResult(
                content=[
                    TextContent(
                        type="text", text=json.dumps([r.model_dump() for r in result])
                    )
                ],
                structured_content={"result": result},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error creating relations: {e}")
            raise ToolError(f"Neo4j error creating relations: {e}") from e
        except Exception as e:
            logger.error(f"Error creating relations: {e}")
            raise ToolError(f"Error creating relations: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "add_feedback",
        annotations=ToolAnnotations(
            title="Add Feedback",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def add_feedback(
        source: str,
        target: str,
        relationType: str,
        sentiment: float,
        asserted_by: str = "System",

    ) -> ToolResult:
        """Apply feedback to a relationship to adjust its weight.

        Sentiment should be between -5.0 and 5.0.
        Positive sentiment increases weight, negative decreases it.
        """
        logger.info(
            f"MCP tool: add_feedback ({relationType} from {source} to {target}, kwargs: {kwargs})"
        )
        try:
            feedback = Feedback(
                source=source,
                target=target,
                relationType=relationType,
                sentiment=sentiment,
                asserted_by=asserted_by,
            )
            result = await memory.add_feedback(feedback)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content=result,
            )
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            raise ToolError(f"Error adding feedback: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "add_observations",
        annotations=ToolAnnotations(
            title="Add Observations",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def add_observations(
        observations: list[ObservationAddition] = Field(
            ..., description="List of observations to add to existing entities"
        ),

    ) -> ToolResult:
        """Add new observations/facts to existing entities in the knowledge graph.

        Appends new observations as unique nodes linked via SUPERSEDES to track temporal changes.
        Use 'asserted_by' to define the persona adding the fact.

        Returns:
            list[dict]: Details about the added observations including entity name and new facts

        Example call:
        {
            "observations": [
                {
                    "entityName": "Alice Johnson",
                    "observations": ["Promoted to Senior Engineer", "Completed AWS certification"],
                    "asserted_by": "The Coach"
                }
            ]
        }
        """
        logger.info(
            f"MCP tool: add_observations ({len(observations)} additions, kwargs: {kwargs})"
        )
        try:
            observation_objects = [
                ObservationAddition.model_validate(obs) for obs in observations
            ]
            result = await memory.add_observations(observation_objects)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content={"result": result},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error adding observations: {e}")
            raise ToolError(f"Neo4j error adding observations: {e}") from e
        except Exception as e:
            logger.error(f"Error adding observations: {e}")
            raise ToolError(f"Error adding observations: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "delete_entities",
        annotations=ToolAnnotations(
            title="Delete Entities",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def delete_entities(
        entityNames: list[str] = Field(
            ..., description="List of exact entity names to delete (deprecate)"
        ),

    ) -> ToolResult:
        """Mark entities and their associated observations as deprecated/deleted.

        Following the Axiom of Non-Erasure, this does not actually destroy the node, but appends
        a deprecation observation so it is excluded from normal queries.

        Returns:
            str: Success confirmation message

        Example call:
        {
            "entityNames": ["Old Company", "Outdated Person"]
        }
        """
        logger.info(
            f"MCP tool: delete_entities ({len(entityNames)} entities, kwargs: {kwargs})"
        )
        try:
            await memory.delete_entities(entityNames)
            return ToolResult(
                content=[
                    TextContent(type="text", text="Entities deprecated successfully")
                ],
                structured_content={"result": "Entities deprecated successfully"},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting entities: {e}")
            raise ToolError(f"Neo4j error deleting entities: {e}") from e
        except Exception as e:
            logger.error(f"Error deleting entities: {e}")
            raise ToolError(f"Error deleting entities: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "delete_observations",
        annotations=ToolAnnotations(
            title="Delete Observations",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def delete_observations(
        deletions: list[ObservationDeletion] = Field(
            ..., description="List of specific observations to deprecate from entities"
        ),

    ) -> ToolResult:
        """Mark specific observations as deleted/deprecated.

        Following the Axiom of Non-Erasure, the observation is flagged as deleted rather than destroyed.

        Returns:
            str: Success confirmation message

        Example call:
        {
            "deletions": [
                {
                    "entityName": "Alice Johnson",
                    "observations": ["Old job title", "Outdated phone number"],
                    "asserted_by": "System"
                }
            ]
        }
        """
        logger.info(
            f"MCP tool: delete_observations ({len(deletions)} deletions, kwargs: {kwargs})"
        )
        try:
            deletion_objects = [
                ObservationDeletion.model_validate(deletion) for deletion in deletions
            ]
            await memory.delete_observations(deletion_objects)
            return ToolResult(
                content=[
                    TextContent(
                        type="text", text="Observations deprecated successfully"
                    )
                ],
                structured_content={"result": "Observations deprecated successfully"},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting observations: {e}")
            raise ToolError(f"Neo4j error deleting observations: {e}") from e
        except Exception as e:
            logger.error(f"Error deleting observations: {e}")
            raise ToolError(f"Error deleting observations: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "delete_relations",
        annotations=ToolAnnotations(
            title="Delete Relations",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def delete_relations(
        relations: list[Relation] = Field(
            ..., description="List of specific relationships to delete from the graph"
        ),

    ) -> ToolResult:
        """Delete specific relationships between entities in the knowledge graph.

        Removes relationships while keeping the entities themselves. The source, target, and
        relationship type must match exactly for deletion. This only affects the relationships,
        not the entities they connect.

        Returns:
            str: Success confirmation message

        Example call:
        {
            "relations": [
                {
                    "source": "Alice Johnson",
                    "target": "Old Company",
                    "relationType": "WORKS_AT"
                }
            ]
        }

        Note: All fields (source, target, relationType) must match exactly for deletion.
        """
        logger.info(
            f"MCP tool: delete_relations ({len(relations)} relations, kwargs: {kwargs})"
        )
        try:
            relation_objects = [
                Relation.model_validate(relation) for relation in relations
            ]
            await memory.delete_relations(relation_objects)
            return ToolResult(
                content=[
                    TextContent(type="text", text="Relations deleted successfully")
                ],
                structured_content={"result": "Relations deleted successfully"},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting relations: {e}")
            raise ToolError(f"Neo4j error deleting relations: {e}") from e
        except Exception as e:
            logger.error(f"Error deleting relations: {e}")
            raise ToolError(f"Error deleting relations: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "search_memories",
        annotations=ToolAnnotations(
            title="Search Memories",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def search_memories(
        query: str = Field(..., description="Hybrid search query (Vector + Fulltext)"),
        top_k: int = Field(
            5, description="Number of entities to return (5 for speed, 25 for depth)"
        ),
        include_hop: bool = Field(
            False,
            description="Whether to include 1-hop relationships (increases token usage)",
        ),
        node_set: str | None = Field(
            None, description="Optional: Filter by Node Set (Memory Domain)"
        ),
        entity_type: str | None = Field(
            None, description="Optional: Filter by Entity Type (e.g., 'person')"
        ),

    ) -> ToolResult:
        """Search for entities in the knowledge graph using Hybrid Search (Vector + Fulltext).

        This tool combines semantic vector similarity with traditional keyword matching across
        names, types, and observations.

        Scoping:
        - top_k=5, include_hop=False: High-speed discovery, minimal tokens.
        - top_k=25, include_hop=True: Deep research, full neighborhood context.

        Returns:
            KnowledgeGraph: Subgraph containing matching entities and their relationships
        """
        logger.info(
            f"MCP tool: search_memories ('{query}', top_k={top_k}, hop={include_hop}, node_set={node_set}, type={entity_type}, kwargs: {kwargs})"
        )
        try:
            result = await memory.search_memories(
                query,
                top_k=top_k,
                include_hop=include_hop,
                node_set=node_set,
                entity_type=entity_type,
            )
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error searching memories: {e}")
            raise ToolError(f"Neo4j error searching memories: {e}") from e
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise ToolError(f"Error searching memories: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "find_memories_by_name",
        annotations=ToolAnnotations(
            title="Find Memories by Name",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def find_memories_by_name(
        names: list[str] = Field(
            ..., description="List of exact entity names to retrieve"
        ),

    ) -> ToolResult:
        """Find specific entities by their exact names.

        Retrieves entities that exactly match the provided names, along with all their
        relationships and connected entities. Use this when you know the exact entity names.

        Returns:
            KnowledgeGraph: Subgraph containing the specified entities and their relationships

        Example call:
        {
            "names": ["Alice Johnson", "Microsoft", "Seattle"]
        }

        This retrieves the entities with exactly those names plus their connections.
        """
        logger.info(
            f"MCP tool: find_memories_by_name ({len(names)} names, kwargs: {kwargs})"
        )
        try:
            result = await memory.find_memories_by_name(names)
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error finding memories by name: {e}")
            raise ToolError(f"Neo4j error finding memories by name: {e}") from e
        except Exception as e:
            logger.error(f"Error finding memories by name: {e}")
            raise ToolError(f"Error finding memories by name: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "rebalance_graph",
        annotations=ToolAnnotations(
            title="Rebalance Graph",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def rebalance_graph() -> ToolResult:
        """Recalculate impact scores for all entities in the graph.

        Uses the Cognitive Impact Scoring (CIS) formula:
        CIS = (utility * log10(frequency + 1)) * exp(-lambda * age)

        This tool should be run periodically (e.g., during the Janitor cycle)
        to ensure memory priority reflects recent activity and time decay.

        Returns:
            str: Summary of the rebalancing operation
        """
        logger.info(f"MCP tool: rebalance_graph")
        try:
            count = await memory.rebalance_graph()
            return ToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Successfully rebalanced {count} entities."
                    )
                ],
                structured_content={"count": count},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error rebalancing graph: {e}")
            raise ToolError(f"Neo4j error rebalancing graph: {e}") from e
        except Exception as e:
            logger.error(f"Error rebalancing graph: {e}")
            raise ToolError(f"Error rebalancing graph: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "sync_labels",
        annotations=ToolAnnotations(
            title="Sync Labels",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def sync_labels() -> ToolResult:
        """One-time migration to ensure all entities have labels matching their 'type' property.

        This is required for the tiered decay model to function correctly, as it relies on
        Neo4j node labels (e.g., :Project, :Task) for heuristic weighting.

        Returns:
            str: Summary of the sync operation
        """
        logger.info(f"MCP tool: sync_labels")
        try:
            count = await memory.sync_labels()
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully synced labels for {count} entities.",
                    )
                ],
                structured_content={"count": count},
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error syncing labels: {e}")
            raise ToolError(f"Neo4j error syncing labels: {e}") from e
        except Exception as e:
            logger.error(f"Error syncing labels: {e}")
            raise ToolError(f"Error syncing labels: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "cognify_domain",
        annotations=ToolAnnotations(
            title="Cognify Domain",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def cognify_domain(
        node_set: str = Field(
            ..., description="The Node Set (Memory Domain) to summarize"
        ),
        summary: str = Field(
            ..., description="The summary text for this domain's history"
        ),
        entityName: str = Field(
            None, description="Optional: Filter summarization to a specific entity"
        ),
        asserted_by: str = Field(
            "System", description="The Persona asserting the summary"
        ),

    ) -> ToolResult:
        """Simplified 'Cognify' step: Summarize a domain's history and supersede existing logs.

        This tool creates a single summary observation that SUPERSEDES all current leaf
        observations in the specified node_set (optionally filtered by entity). Use this
        to compress long observation histories into high-density summaries.

        Returns:
            dict: Status and whether a summary was created

        Example call:
        {
            "node_set": "Katiskankatu",
            "summary": "Project Katiskankatu is currently in the planning phase. Major milestones include...",
            "entityName": "Taloprojekti Katiskankatu 21"
        }
        """
        logger.info(
            f"MCP tool: cognify_domain (node_set={node_set}, entity={entityName}, kwargs: {kwargs})"
        )
        try:
            result = await memory.cognify_domain(
                node_set, summary, entityName, asserted_by
            )
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content=result,
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error during cognify: {e}")
            raise ToolError(f"Neo4j error during cognify: {e}") from e
        except Exception as e:
            logger.error(f"Error during cognify: {e}")
            raise ToolError(f"Error during cognify: {e}") from e

    @mcp.tool(
        name=namespace_prefix + "vectorize_entities",
        annotations=ToolAnnotations(
            title="Vectorize Entities",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def vectorize_entities(
        names: list[str] = Field(
            None, description="Optional: List of exact entity names to vectorize"
        ),
        limit: int = Field(
            100,
            description="Optional: Maximum number of entities to process if names not provided",
        ),

    ) -> ToolResult:
        """Connect a production-ready vectorization engine to generate embeddings for Neo4j entities.

        This tool generates semantic embeddings for entities using the configured VectorEngine
        (e.g., Gemini or Ollama). It combines the entity name, type, and all active observations
        to create a high-density semantic representation.

        Returns:
            str: Summary of the vectorization operation
        """
        logger.info(
            f"MCP tool: vectorize_entities (names={names}, limit={limit}, kwargs: {kwargs})"
        )
        try:
            count = await memory.vectorize_entities(names=names, limit=limit)
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully vectorized {count} observations.",
                    )
                ],
                structured_content={"count": count},
            )
        except Exception as e:
            logger.error(f"Error during vectorization: {e}")
            raise ToolError(f"Error during vectorization: {e}") from e

    return mcp


async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    gemini_api_key: str | None = None,
    ollama_base_url: str | None = None,
    ollama_model: str = "nomic-embed-text",
    transport: Literal["stdio", "sse", "http"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    allow_origins: list[str] = None,
    allowed_hosts: list[str] = None,
) -> None:
    if allowed_hosts is None:
        allowed_hosts = []
    if allow_origins is None:
        allow_origins = []
    logger.info("Starting Neo4j MCP Memory Server")
    logger.info(f"Connecting to Neo4j with DB URL: {neo4j_uri}")

    # Connect to Neo4j
    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password), database=neo4j_database
    )

    # Verify connection
    try:
        await neo4j_driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        exit(1)

    # Initialize vector engine
    vector_engine = None
    if gemini_api_key:
        vector_engine = GeminiVectorEngine(gemini_api_key)
        logger.info("GeminiVectorEngine initialized")
    elif ollama_base_url:
        vector_engine = OllamaVectorEngine(ollama_base_url, ollama_model)
        logger.info(f"OllamaVectorEngine initialized ({ollama_model})")

    # Initialize memory
    memory = Neo4jMemory(neo4j_driver, vector_engine)
    logger.info("Neo4jMemory initialized")

    # Create fulltext index
    await memory.create_fulltext_index()

    # Configure security middleware
    custom_middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        ),
        Middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts),
    ]

    # Create MCP server
    mcp = create_mcp_server(memory, namespace)
    logger.info("MCP server created")

    # Run the server with the specified transport
    logger.info(f"Starting server with transport: {transport}")
    match transport:
        case "http":
            logger.info(f"HTTP server starting on {host}:{port}{path}")
            await mcp.run_http_async(
                host=host, port=port, path=path, middleware=custom_middleware
            )
        case "stdio":
            logger.info("STDIO server starting")
            await mcp.run_stdio_async(show_banner=False)
        case "sse":
            logger.info(f"SSE server starting on {host}:{port}{path}")
            await mcp.run_http_async(
                host=host,
                port=port,
                path=path,
                middleware=custom_middleware,
                transport="sse",
            )
        case _:
            raise ValueError(f"Unsupported transport: {transport}")
