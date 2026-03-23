import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from neo4j import AsyncDriver, RoutingControl
from pydantic import BaseModel, Field

from .vector_engine import VectorEngine

# Set up logging
logger = logging.getLogger('mcp_neo4j_memory')
logger.setLevel(logging.INFO)

class Observation(BaseModel):
    content: str = Field(
        description="The actual fact, event, or state observed.",
        min_length=1
    )
    valid_time: str = Field(
        description="ISO8601 timestamp of when the event actually occurred in reality.",
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    transaction_time: str = Field(
        description="ISO8601 timestamp of when the event was recorded in the database.",
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    asserted_by: str = Field(
        description="The Persona or Agent that recorded this observation (e.g., 'Global Architect', 'Head Chef').",
        default="System"
    )
    node_set: Optional[str] = Field(
        description="The Node Set (Memory Domain) this observation belongs to.",
        default=None
    )

class Entity(BaseModel):
    """Represents a stateless memory entity anchor in the knowledge graph.

    Example:
    {
        "name": "John Smith",
        "type": "person", 
        "observations": ["Works at Neo4j", "Lives in San Francisco", "Expert in graph databases"],
        "utility": 0.8
    }
    """
    name: str = Field(
        description="Unique identifier/name for the entity. Should be descriptive and specific.",
        min_length=1,
        examples=["John Smith", "Neo4j Inc", "San Francisco"]
    )
    type: str = Field(
        description="Category or classification of the entity. Common types: 'person', 'company', 'location', 'concept', 'event'",
        min_length=1,
        examples=["person", "company", "location", "concept", "event"],
        pattern=r'^[A-Za-z_][A-Za-z0-9_]*$'
    )
    observations: List[str] = Field(
        description="List of initial facts or observations about this entity. Each observation should be a complete, standalone fact.",
        default=[]
    )
    asserted_by: str = Field(
        description="The Persona asserting these initial observations.",
        default="System"
    )
    utility: float = Field(
        description="Base importance weight (0.0-1.0). Defaults to 0.5.",
        default=0.5,
        ge=0.0,
        le=1.0
    )
    frequency: int = Field(
        description="Number of times this entity has been accessed.",
        default=1
    )
    last_accessed: str = Field(
        description="ISO8601 timestamp of last access.",
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    impact_score: float = Field(
        description="Calculated Cognitive Impact Score (CIS).",
        default=0.0
    )
    node_set: Optional[str] = Field(
        description="The Node Set (Memory Domain) this entity belongs to for isolation.",
        default=None
    )
    memory_domain: Optional[str] = Field(
        description="The cognitive memory domain (Episodic, Semantic, etc.).",
        default=None
    )
    embedding: Optional[List[float]] = Field(
        description="Semantic vector embedding for this entity.",
        default=None
    )

class Relation(BaseModel):
    """Represents a relationship between two entities in the knowledge graph.
    
    Example:
    {
        "source": "John Smith",
        "target": "Neo4j Inc", 
        "relationType": "WORKS_AT",
        "weight": 1.2
    }
    """
    source: str = Field(
        description="Name of the source entity (must match an existing entity name exactly)",
        min_length=1,
        examples=["John Smith", "Neo4j Inc"]
    )
    target: str = Field(
        description="Name of the target entity (must match an existing entity name exactly)",
        min_length=1, 
        examples=["Neo4j Inc", "San Francisco"]
    )
    relationType: str = Field(
        description="Type of relationship between source and target. Use descriptive, uppercase names with underscores.",
        min_length=1,
        examples=["WORKS_AT", "LIVES_IN", "MANAGES", "COLLABORATES_WITH", "LOCATED_IN"],
        pattern=r'^[A-Za-z_][A-Za-z0-9_]*$'
    )
    weight: float = Field(
        description="The weight or strength of the relationship, optimized via feedback loops.",
        default=1.0
    )

class Feedback(BaseModel):
    """Agent or user feedback for a specific relationship to optimize retrieval weights."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relationType: str = Field(description="Relation type")
    sentiment: float = Field(
        description="Sentiment score (-5.0 to 5.0) to adjust edge weight.",
        ge=-5.0, 
        le=5.0
    )
    asserted_by: str = Field(description="Persona providing feedback", default="System")

class KnowledgeGraph(BaseModel):
    """Complete knowledge graph containing entities and their relationships."""
    entities: List[Entity] = Field(
        description="List of all entities in the knowledge graph",
        default=[]
    )
    relations: List[Relation] = Field(
        description="List of all relationships between entities",
        default=[]
    )

class ObservationAddition(BaseModel):
    """Request to add new observations to an existing entity.
    
    Example:
    {
        "entityName": "John Smith",
        "observations": ["Recently promoted to Senior Engineer", "Speaks fluent German"],
        "asserted_by": "Chief of Staff"
    }
    """
    entityName: str = Field(
        description="Exact name of the existing entity to add observations to",
        min_length=1,
        examples=["John Smith", "Neo4j Inc"]
    )
    observations: List[str] = Field(
        description="New observations/facts to add to the entity. Each should be unique and informative.",
        min_length=1
    )
    asserted_by: str = Field(
        description="The Persona asserting these observations.",
        default="System"
    )
    valid_time: Optional[str] = Field(
        description="Optional ISO8601 timestamp of when these events actually occurred. Defaults to now.",
        default=None
    )
    node_set: Optional[str] = Field(
        description="Optional Node Set to link these observations to.",
        default=None
    )

class ObservationDeletion(BaseModel):
    """Request to mark specific observations as deprecated/deleted."""
    entityName: str = Field(
        description="Exact name of the existing entity to remove observations from",
        min_length=1,
        examples=["John Smith", "Neo4j Inc"]
    )
    observations: List[str] = Field(
        description="Exact observation texts to delete/deprecate from the entity",
        min_length=1
    )
    asserted_by: str = Field(
        description="The Persona asserting the deletion.",
        default="System"
    )

class Neo4jMemory:
    def __init__(self, neo4j_driver: AsyncDriver, vector_engine: Optional[VectorEngine] = None):
        self.driver = neo4j_driver
        self.vector_engine = vector_engine

    async def create_fulltext_index(self):
        """Create fulltext search indices for entities and observations if they don't exist."""
        try:
            query_entity = "CREATE FULLTEXT INDEX search_entity IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.type];"
            await self.driver.execute_query(query_entity, routing_control=RoutingControl.WRITE)
            query_obs = "CREATE FULLTEXT INDEX search_obs IF NOT EXISTS FOR (o:Observation) ON EACH [o.content];"
            await self.driver.execute_query(query_obs, routing_control=RoutingControl.WRITE)
            logger.info("Created fulltext search indices")
        except Exception as e:
            # Index might already exist, which is fine
            logger.debug(f"Fulltext index creation: {e}")

    async def load_graph(self, filter_query: str = "*"):
        """Load the entire knowledge graph from Neo4j."""
        logger.info("Loading knowledge graph from Neo4j")
        
        if filter_query == "*":
            query = """
            MATCH (entity:Entity)
            OPTIONAL MATCH (o:Observation)-[:OBSERVES_STATE]->(entity)
            WHERE o.is_deleted IS NULL OR o.is_deleted = false
            WITH entity, collect(o.content) AS obs
            OPTIONAL MATCH (entity)-[r]->(other:Entity)
            RETURN collect(distinct {
                name: entity.name, 
                type: entity.type, 
                observations: obs,
                utility: coalesce(entity.utility, 0.5),
                frequency: coalesce(entity.frequency, 1),
                last_accessed: coalesce(entity.last_accessed, toString(datetime())),
                impact_score: coalesce(entity.impact_score, 0.0),
                node_set: entity.node_set,
                memory_domain: entity.memory_domain,
                embedding: entity.embedding
            }) as nodes,
            collect(distinct {
                source: startNode(r).name, 
                target: endNode(r).name, 
                relationType: type(r),
                weight: coalesce(r.weight, 1.0)
            }) as relations
            """
            params = {}
        else:
            query = """
            CALL db.index.fulltext.queryNodes('search_entity', $filter) YIELD node as entity, score
            OPTIONAL MATCH (o:Observation)-[:OBSERVES_STATE]->(entity)
            WHERE o.is_deleted IS NULL OR o.is_deleted = false
            WITH entity, collect(o.content) AS obs
            OPTIONAL MATCH (entity)-[r]->(other:Entity)
            RETURN collect(distinct {
                name: entity.name, 
                type: entity.type, 
                observations: obs,
                utility: coalesce(entity.utility, 0.5),
                frequency: coalesce(entity.frequency, 1),
                last_accessed: coalesce(entity.last_accessed, toString(datetime())),
                impact_score: coalesce(entity.impact_score, 0.0),
                node_set: entity.node_set,
                memory_domain: entity.memory_domain,
                embedding: entity.embedding
            }) as nodes,
            collect(distinct {
                source: startNode(r).name, 
                target: endNode(r).name, 
                relationType: type(r),
                weight: coalesce(r.weight, 1.0)
            }) as relations
            """
            params = {"filter": filter_query}
        
        result = await self.driver.execute_query(query, params, routing_control=RoutingControl.READ)
        
        if not result.records:
            return KnowledgeGraph(entities=[], relations=[])
        
        record = result.records[0]
        nodes = record.get('nodes', list())
        rels = record.get('relations', list())
        
        entities = [
            Entity(
                name=node['name'],
                type=node['type'],
                observations=node.get('observations', list()),
                utility=node['utility'],
                frequency=node['frequency'],
                last_accessed=node['last_accessed'],
                impact_score=node['impact_score'],
                node_set=node.get('node_set'),
                memory_domain=node.get('memory_domain'),
                embedding=node.get('embedding')
            )
            for node in nodes if node.get('name')
        ]
        
        relations = [
            Relation(
                source=rel['source'],
                target=rel['target'],
                relationType=rel['relationType'],
                weight=rel.get('weight', 1.0)
            )
            for rel in rels if rel.get('relationType')
        ]
        
        logger.debug(f"Loaded entities: {entities}")
        logger.debug(f"Loaded relations: {relations}")
        
        return KnowledgeGraph(entities=entities, relations=relations)

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities in the knowledge graph, along with their initial observations."""
        logger.info(f"Creating {len(entities)} entities")
        for entity in entities:
            # 1. Create the Entity (stateless anchor)
            # 2. Ensure Persona exists
            # 3. Create initial Observations and link them to Entity
            # 4. Apply dynamic labels based on 'type' property (standard Cypher)
            # 5. Link to Node Set (Memory Domain) if provided
            query = f"""
            WITH $entity as entity
            MERGE (e:Entity {{ name: entity.name }})
            ON CREATE SET e.type = entity.type, 
                          e.created_at = datetime(),
                          e.utility = coalesce(entity.utility, 0.5),
                          e.frequency = 1,
                          e.last_accessed = toString(datetime()),
                          e.impact_score = 0.0,
                          e.node_set = entity.node_set,
                          e.memory_domain = entity.memory_domain,
                          e.embedding = entity.embedding
            ON MATCH SET  e.utility = coalesce(entity.utility, e.utility),
                          e.node_set = coalesce(entity.node_set, e.node_set),
                          e.memory_domain = coalesce(entity.memory_domain, e.memory_domain),
                          e.embedding = coalesce(entity.embedding, e.embedding)
            
            // Link to Node Set node
            FOREACH (ignore IN CASE WHEN entity.node_set IS NOT NULL THEN [1] ELSE [] END |
                MERGE (ns:NodeSet {{ name: entity.node_set }})
                MERGE (e)-[:BELONGS_TO_SET]->(ns)
            )

            // Dynamic Label Application via FOREACH (Standard Cypher workaround)
            FOREACH (ignore IN CASE WHEN entity.type = 'Project' THEN [1] ELSE [] END | SET e:Project)
            FOREACH (ignore IN CASE WHEN entity.type = 'Infrastructure' THEN [1] ELSE [] END | SET e:Infrastructure)
            FOREACH (ignore IN CASE WHEN entity.type = 'Regulation' THEN [1] ELSE [] END | SET e:Regulation)
            FOREACH (ignore IN CASE WHEN entity.type = 'Knowledge' THEN [1] ELSE [] END | SET e:Knowledge)
            FOREACH (ignore IN CASE WHEN entity.type = 'Fact' THEN [1] ELSE [] END | SET e:Fact)
            FOREACH (ignore IN CASE WHEN entity.type = 'Task' THEN [1] ELSE [] END | SET e:Task)
            FOREACH (ignore IN CASE WHEN entity.type = 'Personal' THEN [1] ELSE [] END | SET e:Personal)
            FOREACH (ignore IN CASE WHEN entity.type = 'Observation' THEN [1] ELSE [] END | SET e:Observation)
            FOREACH (ignore IN CASE WHEN entity.type = 'Event' THEN [1] ELSE [] END | SET e:Event)
            FOREACH (ignore IN CASE WHEN entity.type = 'Audit' THEN [1] ELSE [] END | SET e:Audit)

            MERGE (p:Persona {{ name: entity.asserted_by }})
            
            WITH e, p, entity.observations AS obs_list, entity.node_set AS ns_name
            UNWIND obs_list AS obs_text
            
            CREATE (o:Observation {{ 
                content: obs_text,
                valid_time: toString(datetime()),
                transaction_time: toString(datetime()),
                node_set: ns_name
            }})
            MERGE (o)-[:OBSERVES_STATE]->(e)
            MERGE (p)-[:ASSERTED_BY]->(o)
            
            FOREACH (ignore IN CASE WHEN ns_name IS NOT NULL THEN [1] ELSE [] END |
                MERGE (ns:NodeSet {{ name: ns_name }})
                MERGE (o)-[:BELONGS_TO_SET]->(ns)
            )
            """
            await self.driver.execute_query(query, {"entity": entity.model_dump()}, routing_control=RoutingControl.WRITE)
            
            # Auto-vectorize if embedding is missing
            if self.vector_engine and not entity.embedding:
                await self._auto_vectorize(entity.name)

        return entities

    async def sync_labels(self) -> int:
        """One-time migration to ensure all entities have labels matching their 'type' property."""
        logger.info("Syncing node labels with 'type' properties for CIS tiered decay")
        query = """
        MATCH (e:Entity)
        WHERE e.type IS NOT NULL
        FOREACH (ignore IN CASE WHEN e.type = 'Project' THEN [1] ELSE [] END | SET e:Project)
        FOREACH (ignore IN CASE WHEN e.type = 'Infrastructure' THEN [1] ELSE [] END | SET e:Infrastructure)
        FOREACH (ignore IN CASE WHEN e.type = 'Regulation' THEN [1] ELSE [] END | SET e:Regulation)
        FOREACH (ignore IN CASE WHEN e.type = 'Knowledge' THEN [1] ELSE [] END | SET e:Knowledge)
        FOREACH (ignore IN CASE WHEN e.type = 'Fact' THEN [1] ELSE [] END | SET e:Fact)
        FOREACH (ignore IN CASE WHEN e.type = 'Task' THEN [1] ELSE [] END | SET e:Task)
        FOREACH (ignore IN CASE WHEN e.type = 'Personal' THEN [1] ELSE [] END | SET e:Personal)
        FOREACH (ignore IN CASE WHEN e.type = 'Observation' THEN [1] ELSE [] END | SET e:Observation)
        FOREACH (ignore IN CASE WHEN e.type = 'Event' THEN [1] ELSE [] END | SET e:Event)
        FOREACH (ignore IN CASE WHEN e.type = 'Audit' THEN [1] ELSE [] END | SET e:Audit)
        RETURN count(e) as count
        """
        result = await self.driver.execute_query(query, routing_control=RoutingControl.WRITE)
        count = result.records[0].get("count", 0)
        logger.info(f"Synced labels for {count} entities.")
        return count

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations between entities with weight support."""
        logger.info(f"Creating {len(relations)} relations")
        for relation in relations:
            query = f"""
            WITH $relation as relation
            MATCH (from:Entity),(to:Entity)
            WHERE from.name = relation.source
            AND  to.name = relation.target
            MERGE (from)-[r:`{relation.relationType}`]->(to)
            ON CREATE SET r.weight = coalesce(relation.weight, 1.0)
            ON MATCH SET r.weight = coalesce(relation.weight, r.weight)
            """
            
            await self.driver.execute_query(
                query, 
                {"relation": relation.model_dump()},
                routing_control=RoutingControl.WRITE
            )

        return relations

    async def add_feedback(self, feedback: Feedback) -> Dict[str, Any]:
        """Apply agent feedback to adjust relationship weights (Feedback Loop)."""
        logger.info(f"Applying feedback for {feedback.relationType} from {feedback.source} to {feedback.target}")
        query = f"MATCH (source:Entity {{ name: $source }})-[r:`{feedback.relationType}`]->(target:Entity {{ name: $target }}) SET r.weight = coalesce(r.weight, 1.0) + ($sentiment * 0.1) RETURN r.weight as new_weight"
        result = await self.driver.execute_query(
            query,
            {
                "source": feedback.source,
                "target": feedback.target,
                "sentiment": feedback.sentiment
            },
            routing_control=RoutingControl.WRITE
        )
        
        if not result.records:
            return {"status": "failed", "message": "Relation not found"}
            
        return {"status": "success", "new_weight": result.records[0].get("new_weight")}

    async def add_observations(self, observations: List[ObservationAddition]) -> List[Dict[str, Any]]:
        """Add new observations to existing entities following the Graph-as-Log event sourcing model."""
        logger.info(f"Adding observations to {len(observations)} entities")
        
        results = []
        for obs in observations:
            query = """
            MATCH (e:Entity { name: $entityName })
            MERGE (p:Persona { name: $asserted_by })
            
            WITH e, p
            UNWIND $observations AS obs_text
            
            // 1. Create the new observation first
            CREATE (new_o:Observation {
                content: obs_text,
                valid_time: coalesce($valid_time, toString(datetime())),
                transaction_time: toString(datetime()),
                node_set: $node_set
            })
            MERGE (new_o)-[:OBSERVES_STATE]->(e)
            MERGE (p)-[:ASSERTED_BY]->(new_o)
            
            // 2. Conditional link to NodeSet node
            FOREACH (ns_name IN CASE WHEN $node_set IS NOT NULL THEN [$node_set] ELSE [] END |
                MERGE (ns:NodeSet { name: ns_name })
                MERGE (new_o)-[:BELONGS_TO_SET]->(ns)
            )
            
            // 3. Find the *current* latest WITHIN THE SAME NODE_SET to chain properly
            // CRITICAL: Exclude the new_o node itself
            WITH e, new_o, obs_text
            OPTIONAL MATCH (e)<-[:OBSERVES_STATE]-(candidate:Observation)
            WHERE (candidate.node_set = new_o.node_set OR (candidate.node_set IS NULL AND new_o.node_set IS NULL))
            AND NOT ()-[:SUPERSEDES]->(candidate)
            AND id(candidate) <> id(new_o)
            
            WITH e, new_o, obs_text, candidate
            ORDER BY candidate.transaction_time DESC
            WITH e, new_o, obs_text, collect(candidate)[0] AS latest
            
            FOREACH (x IN CASE WHEN latest IS NOT NULL THEN [latest] ELSE [] END |
                MERGE (new_o)-[:SUPERSEDES]->(x)
                MERGE (new_o)-[:SAME_CC_AS]->(x)
            )
            
            RETURN e.name as name, obs_text as new
            """
            
            result = await self.driver.execute_query(
                query, 
                {
                    "entityName": obs.entityName,
                    "asserted_by": obs.asserted_by,
                    "observations": obs.observations,
                    "valid_time": obs.valid_time,
                    "node_set": obs.node_set
                },
                routing_control=RoutingControl.WRITE
            )
            
            added = [record.get("new") for record in result.records]
            if added:
                results.append({"entityName": obs.entityName, "addedObservations": added})
                
                # Re-vectorize entity to reflect new observations
                if self.vector_engine:
                    await self._auto_vectorize(obs.entityName)

        return results

    async def cognify_domain(self, node_set: str, summary: str, entity_name: Optional[str] = None, asserted_by: str = "System") -> Dict[str, Any]:
        """Simplified 'Cognify' step: Summarize a domain's history and supersede existing logs.
        
        This tool creates a single summary observation that SUPERSEDES all current leaf 
        observations in the specified node_set (optionally filtered by entity).
        """
        logger.info(f"Cognifying domain '{node_set}'" + (f" for entity '{entity_name}'" if entity_name else ""))
        
        query = """
        // 1. Find all leaf observations in the domain
        MATCH (o:Observation)
        WHERE (o.node_set = $node_set)
        AND NOT ()-[:SUPERSEDES]->(o)
        """
        
        if entity_name:
            query += " AND (o)-[:OBSERVES_STATE]->(:Entity { name: $entityName })"
            
        query += """
        WITH collect(o) as leaves
        WHERE size(leaves) > 0
        
        // 2. Create the summary observation
        CREATE (summary:Observation {
            content: $summary,
            valid_time: toString(datetime()),
            transaction_time: toString(datetime()),
            node_set: $node_set,
            is_summary: true
        })
        
        // 3. Link summary to persona
        MERGE (p:Persona { name: $asserted_by })
        MERGE (p)-[:ASSERTED_BY]->(summary)
        
        // 4. Supersede all leaves and link to their entities
        WITH summary, leaves
        UNWIND leaves AS leaf
        MERGE (summary)-[:SUPERSEDES]->(leaf)
        WITH summary, leaf
        MATCH (leaf)-[:OBSERVES_STATE]->(e:Entity)
        MERGE (summary)-[:OBSERVES_STATE]->(e)
        
        // 5. Link to Node Set node
        WITH summary
        MERGE (ns:NodeSet { name: $node_set })
        MERGE (summary)-[:BELONGS_TO_SET]->(ns)
        
        RETURN count(summary) as created
        """
        
        result = await self.driver.execute_query(
            query,
            {
                "node_set": node_set,
                "summary": summary,
                "entityName": entity_name,
                "asserted_by": asserted_by
            },
            routing_control=RoutingControl.WRITE
        )
        
        created = result.records[0].get("created", 0) if result.records else 0
        return {
            "status": "success" if created > 0 else "no_observations_to_summarize",
            "node_set": node_set,
            "summary_created": created > 0
        }

    async def delete_entities(self, entity_names: List[str]) -> None:
        """Deprecate multiple entities instead of actual deletion (Axiom of Non-Erasure)."""
        logger.info(f"Deprecating {len(entity_names)} entities")
        for name in entity_names:
            query = """
            MATCH (e:Entity { name: $name })
            MERGE (p:Persona { name: 'System' })
            
            CALL {
                WITH e
                OPTIONAL MATCH (latest:Observation)-[:OBSERVES_STATE]->(e)
                WHERE NOT ()-[:SUPERSEDES]->(latest)
                RETURN latest
                ORDER BY latest.transaction_time DESC
                LIMIT 1
            }
            
            CREATE (new_o:Observation {
                content: "Entity marked as deleted/deprecated.",
                valid_time: toString(datetime()),
                transaction_time: toString(datetime()),
                is_deleted: true
            })
            MERGE (new_o)-[:OBSERVES_STATE]->(e)
            MERGE (p)-[:ASSERTED_BY]->(new_o)
            
            FOREACH (ignore IN CASE WHEN latest IS NOT NULL THEN [1] ELSE [] END |
                MERGE (new_o)-[:SUPERSEDES]->(latest)
                MERGE (new_o)-[:SAME_CC_AS]->(latest)
            )
            """
            await self.driver.execute_query(query, {"name": name}, routing_control=RoutingControl.WRITE)
        logger.info(f"Successfully deprecated {len(entity_names)} entities")

    async def delete_observations(self, deletions: List[ObservationDeletion]) -> None:
        """Deprecate specific observations from entities."""
        logger.info(f"Deprecating observations from {len(deletions)} entities")
        for d in deletions:
            query = """
            MATCH (e:Entity { name: $entityName })<-[:OBSERVES_STATE]-(o:Observation)
            WHERE o.content IN $observations
            SET o.is_deleted = true
            """
            await self.driver.execute_query(
                query, 
                {"entityName": d.entityName, "observations": d.observations},
                routing_control=RoutingControl.WRITE
            )
        logger.info(f"Successfully deprecated observations from {len(deletions)} entities")

    async def delete_relations(self, relations: List[Relation]) -> None:
        """Delete multiple relations from the graph."""
        logger.info(f"Deleting {len(relations)} relations")
        for relation in relations:
            query = f"""
            WITH $relation as relation
            MATCH (source:Entity)-[r:`{relation.relationType}`]->(target:Entity)
            WHERE source.name = relation.source
            AND target.name = relation.target
            DELETE r
            """
            await self.driver.execute_query(
                query, 
                {"relation": relation.model_dump()},
                routing_control=RoutingControl.WRITE
            )
        logger.info(f"Successfully deleted {len(relations)} relations")

    async def track_access(self, names: List[str]) -> None:
        """Increment frequency and update last_accessed for the given entity names."""
        if not names:
            return
        logger.info(f"Tracking access for {len(names)} entities")
        query = """
        UNWIND $names as name
        MATCH (e:Entity { name: name })
        SET e.frequency = coalesce(e.frequency, 0) + 1,
            e.last_accessed = toString(datetime())
        """
        await self.driver.execute_query(query, {"names": names}, routing_control=RoutingControl.WRITE)

    async def rebalance_graph(self) -> int:
        """Recalculate impact scores for all entities in the graph using the CIS 2.0 formula.
        
        Formula: CIS = (U * log10(F + 1) * S) / (A + 1)
        - U: Utility (Tiered Base + Active Boost)
        - F: Frequency
        - S: Semantic Salience (Multiplier for Epistemic Surprise/Tension)
        - A: Age in days (Transaction Time decay)
        """
        logger.info("Rebalancing knowledge graph with CIS 2.0 formula")
        query = """
        MATCH (e:Entity)
        // 1. Identify Tier based on labels (U - Utility Base)
        WITH e, 
             CASE 
               WHEN "Project" IN labels(e) OR "Infrastructure" IN labels(e) THEN {l: 0.0, def_u: 1.0}
               WHEN "Regulation" IN labels(e) OR "Knowledge" IN labels(e) OR "Fact" IN labels(e) THEN {l: 0.000001, def_u: 0.9}
               WHEN "Task" IN labels(e) OR "Personal" IN labels(e) THEN {l: 0.00001, def_u: 0.7}
               WHEN "Observation" IN labels(e) OR "Event" IN labels(e) OR "Audit" IN labels(e) THEN {l: 0.00005, def_u: 0.5}
               ELSE {l: 0.0001, def_u: 0.3}
             END AS tier
        
        // 2. Check for Active Boost (U - Utility Boost)
        OPTIONAL MATCH (e)<-[:REFERENCED_BY]-(t:Entity {type: 'Task'})
        OPTIONAL MATCH (o:Observation {is_deleted: false})-[:OBSERVES_STATE]->(t)
        WHERE o.content CONTAINS "Stack: 26" OR o.content CONTAINS "Stack: 23"
        
        WITH e, tier, CASE WHEN o IS NOT NULL THEN 0.2 ELSE 0.0 END AS boost
        
        // 3. Calculate Semantic Salience (S - Detail Density Multiplier)
        OPTIONAL MATCH (e)<-[obs_rel:OBSERVES_STATE]-(:Observation)
        WITH e, tier, boost, count(obs_rel) AS obs_count
        // S = 1.0 + (1.0 / (obs_count + 1.0)) -> Entities with fewer observations have higher tension/surprise
        WITH e, tier, boost, obs_count, 1.0 + (1.0 / (obs_count + 1.0)) AS salience
        
        // 4. Calculate Age (A - Transaction Time decay in days)
        // We use created_at if available, fallback to last_accessed, then now
        WITH e, tier, boost, obs_count, salience,
             duration.inSeconds(
                 datetime(coalesce(e.created_at, e.last_accessed, toString(datetime()))), 
                 datetime()
             ).seconds / 86400.0 AS age_days
        
        // 5. Update impact_score (CIS 2.0 Formula)
        // CIS = (U * log10(F + 1) * S) / (A + 1)
        SET e.impact_score = (
            (coalesce(e.utility, tier.def_u) + boost) * 
            log10(coalesce(e.frequency, 1) + 1.0) * 
            salience
        ) / (age_days + 1.0)
        
        RETURN count(e) as count
        """
        result = await self.driver.execute_query(query, routing_control=RoutingControl.WRITE)
        count = result.records[0].get("count", 0)
        logger.info(f"Rebalanced {count} entities with CIS 2.0 formula.")
        return count

    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph."""
        return await self.load_graph()

    async def search_memories(self, query: str, top_k: int = 5, include_hop: bool = False) -> KnowledgeGraph:
        """Search for memories with Hybrid Search and configurable scoping (Top-K / Hop)."""
        logger.info(f"Searching for memories (query='{query}', top_k={top_k}, include_hop={include_hop})")
        
        if not self.vector_engine:
            logger.info("No vector engine configured, falling back to fulltext search.")
            return await self.load_graph(query)

        # 1. Get embedding for the query
        embedding = await self.vector_engine.get_embedding(query)
        if not embedding:
            logger.warning("Failed to generate embedding for query, falling back to fulltext.")
            return await self.load_graph(query)

        # 2. Execute Hybrid Cypher Query with Scoping
        # include_hop=True triggers a 1-hop traversal to fetch relations
        cypher = f"""
        CALL {{
            // A. Vector Search on Entity Embeddings
            CALL db.index.vector.queryNodes('entity_vector_index', $limit, $embedding) YIELD node as entity, score
            RETURN entity, score
            
            UNION
            
            // B. Fulltext Search on Entity Metadata
            CALL db.index.fulltext.queryNodes('search_entity', $query) YIELD node as entity, score
            RETURN entity, score
            
            UNION
            
            // C. Fulltext Search on Observations (returning their parent entities)
            CALL db.index.fulltext.queryNodes('search_obs', $query) YIELD node as obs, score
            MATCH (obs)-[:OBSERVES_STATE]->(entity:Entity)
            RETURN entity, score
        }}
        WITH entity, max(score) as combined_score
        ORDER BY combined_score DESC
        LIMIT $limit
        
        // Side Effect: Track Access
        SET entity.frequency = coalesce(entity.frequency, 0) + 1,
            entity.last_accessed = toString(datetime())
        
        WITH entity
            
        // Gather Observations for the results
        OPTIONAL MATCH (o:Observation)-[:OBSERVES_STATE]->(entity)
        WHERE o.is_deleted IS NULL OR o.is_deleted = false
        WITH entity, collect(distinct o.content) as obs
        
        // Traversal: Conditional 1-hop relations
        // If include_hop is false, r will be null
        OPTIONAL MATCH (entity)-[r]-(other:Entity)
        WHERE $include_hop = true
        
        RETURN collect(distinct {{
            name: entity.name, 
            type: entity.type, 
            observations: obs,
            utility: coalesce(entity.utility, 0.5),
            frequency: coalesce(entity.frequency, 1),
            last_accessed: coalesce(entity.last_accessed, toString(datetime())),
            impact_score: coalesce(entity.impact_score, 0.0),
            node_set: entity.node_set,
            memory_domain: entity.memory_domain,
            embedding: entity.embedding
        }}) as nodes,
        collect(distinct {{
            source: startNode(r).name, 
            target: endNode(r).name, 
            relationType: type(r),
            weight: coalesce(r.weight, 1.0)
        }}) as relations
        """
        
        result = await self.driver.execute_query(
            cypher, 
            {
                "embedding": embedding, 
                "query": query, 
                "limit": top_k, 
                "include_hop": include_hop
            }, 
            routing_control=RoutingControl.WRITE
        )
        
        if not result.records:
            return KnowledgeGraph(entities=[], relations=[])
            
        record = result.records[0]
        nodes = record.get('nodes', list())
        rels = record.get('relations', list())
        
        entities = [
            Entity(
                name=node['name'],
                type=node['type'],
                observations=node.get('observations', list()),
                utility=node['utility'],
                frequency=node['frequency'],
                last_accessed=node['last_accessed'],
                impact_score=node['impact_score'],
                node_set=node.get('node_set'),
                memory_domain=node.get('memory_domain'),
                embedding=node['embedding']
            )
            for node in nodes if node.get('name')
        ]
        
        relations = [
            Relation(
                source=rel['source'],
                target=rel['target'],
                relationType=rel['relationType'],
                weight=rel.get('weight', 1.0)
            )
            for rel in rels if rel.get('relationType')
        ]
        
        logger.info(f"Scoped search (top_k={{top_k}}) found {{len(entities)}} entities and {{len(relations)}} relations.")
        return KnowledgeGraph(entities=entities, relations=relations)

    async def find_memories_by_name(self, names: List[str]) -> KnowledgeGraph:
        """Find specific memories by their names. This does not use fulltext search."""
        logger.info(f"Finding {len(names)} memories by name")
        # Side effect: Track access
        await self.track_access(names)
        
        query = """
        MATCH (e:Entity)
        WHERE e.name IN $names
        OPTIONAL MATCH (o:Observation)-[:OBSERVES_STATE]->(e)
        WHERE o.is_deleted IS NULL OR o.is_deleted = false
        RETURN  e.name as name, 
                e.type as type, 
                collect(o.content) as observations,
                coalesce(e.utility, 0.5) as utility,
                coalesce(e.frequency, 1) as frequency,
                coalesce(e.last_accessed, toString(datetime())) as last_accessed,
                coalesce(e.impact_score, 0.0) as impact_score,
                e.node_set as node_set,
                e.memory_domain as memory_domain,
                e.embedding as embedding
        """
        result_nodes = await self.driver.execute_query(query, {"names": names}, routing_control=RoutingControl.READ)
        entities: list[Entity] = list()
        for record in result_nodes.records:
            entities.append(Entity(
                name=record['name'],
                type=record['type'],
                observations=record.get('observations', list()),
                utility=record['utility'],
                frequency=record['frequency'],
                last_accessed=record['last_accessed'],
                impact_score=record['impact_score'],
                node_set=record.get('node_set'),
                memory_domain=record.get('memory_domain'),
                embedding=record.get('embedding')
            ))
        
        # Get relations for found entities
        relations: list[Relation] = list()
        if entities:
            query = """
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE source.name IN $names OR target.name IN $names
            RETURN  source.name as source, 
                    target.name as target, 
                    type(r) as relationType,
                    coalesce(r.weight, 1.0) as weight
            """
            result_relations = await self.driver.execute_query(query, {"names": names}, routing_control=RoutingControl.READ)
            for record in result_relations.records:
                relations.append(Relation(
                    source=record["source"],
                    target=record["target"],
                    relationType=record["relationType"],
                    weight=record.get("weight", 1.0)
                ))
        
        logger.info(f"Found {len(entities)} entities and {len(relations)} relations")
        return KnowledgeGraph(entities=entities, relations=relations)

    async def vectorize_entities(self, names: Optional[List[str]] = None, limit: int = 100) -> int:
        """Bulk update embeddings for entities lacking them, or for specific named entities."""
        if not self.vector_engine:
            logger.warning("Vectorization requested but no VectorEngine configured.")
            return 0
            
        logger.info(f"Vectorizing entities (names={names}, limit={limit})")
        
        if names:
            query = "MATCH (e:Entity) WHERE e.name IN $names RETURN e.name AS name, e.type AS type"
            params = {"names": names}
        else:
            query = "MATCH (e:Entity) WHERE e.embedding IS NULL RETURN e.name AS name, e.type AS type LIMIT $limit"
            params = {"limit": limit}
            
        result = await self.driver.execute_query(query, params, routing_control=RoutingControl.READ)
        entities = [{"name": record["name"], "type": record["type"]} for record in result.records]
        
        updated_count = 0
        for ent in entities:
            try:
                # 1. Gather context for embedding
                text_to_embed = f"Entity: {ent['name']}, Type: {ent['type']}"
                obs_query = "MATCH (e:Entity {name: $name})<-[:OBSERVES_STATE]-(o:Observation) RETURN o.content AS obs"
                obs_res = await self.driver.execute_query(obs_query, {"name": ent['name']}, routing_control=RoutingControl.READ)
                observations = [rec["obs"] for rec in obs_res.records]
                if observations:
                    text_to_embed += " Observations: " + " ".join(observations)
                
                # 2. Call VectorEngine
                emb = await self.vector_engine.get_embedding(text_to_embed)
                if emb:
                    update_query = "MATCH (e:Entity {name: $name}) SET e.embedding = $emb"
                    await self.driver.execute_query(update_query, {"name": ent['name'], "emb": emb}, routing_control=RoutingControl.WRITE)
                    updated_count += 1
            except Exception as e:
                logger.error(f"Error vectorizing entity '{ent['name']}': {e}")
                
        logger.info(f"Successfully vectorized {updated_count} entities.")
        return updated_count

    async def _auto_vectorize(self, entity_name: str):
        """Internal helper for event-driven vectorization."""
        if self.vector_engine:
            await self.vectorize_entities(names=[entity_name])
