// =====================================================================
// schema.cypher — constraints, lookup indexes and vector indexes for the
// CTIConnect snapshot graph and the report layer (Neo4j 5.x).
// Applied by scripts/load_graph.py before any data is written.
// Statements are separated by ';' at line end.
// =====================================================================

// ---------------------------------------------------------------------
// 1. Uniqueness: one node per snapshot identifier
// ---------------------------------------------------------------------
CREATE CONSTRAINT technique_id IF NOT EXISTS FOR (t:AttackTechnique) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT tactic_name IF NOT EXISTS FOR (t:Tactic) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT cwe_id IF NOT EXISTS FOR (w:CWE) REQUIRE w.id IS UNIQUE;
CREATE CONSTRAINT capec_id IF NOT EXISTS FOR (c:CAPEC) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT cve_id IF NOT EXISTS FOR (v:CVE) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT report_id IF NOT EXISTS FOR (r:ThreatReport) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (k:ReportChunk) REQUIRE k.id IS UNIQUE;
CREATE CONSTRAINT behavior_id IF NOT EXISTS FOR (b:ReportBehavior) REQUIRE b.id IS UNIQUE;

// ---------------------------------------------------------------------
// 2. Lookup indexes: title resolution and card attributes
// ---------------------------------------------------------------------
CREATE INDEX technique_title IF NOT EXISTS FOR (t:AttackTechnique) ON (t.title_norm);
CREATE INDEX cwe_title IF NOT EXISTS FOR (w:CWE) ON (w.title_norm);
CREATE INDEX technique_domain IF NOT EXISTS FOR (t:AttackTechnique) ON (t.domain);
CREATE INDEX chunk_report IF NOT EXISTS FOR (k:ReportChunk) ON (k.report_id);

// ---------------------------------------------------------------------
// 3. Vector indexes: text-embedding-3-large, 3,072 dimensions, cosine
//    (Neo4j reports cosine scores as (1 + cos) / 2)
// ---------------------------------------------------------------------
CREATE VECTOR INDEX technique_emb IF NOT EXISTS FOR (t:AttackTechnique) ON (t.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX cwe_emb IF NOT EXISTS FOR (w:CWE) ON (w.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX capec_emb IF NOT EXISTS FOR (c:CAPEC) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX cve_emb IF NOT EXISTS FOR (v:CVE) ON (v.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX behavior_emb IF NOT EXISTS FOR (b:ReportBehavior) ON (b.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};

// ---------------------------------------------------------------------
// 4. Example query: the CAPEC bridge (rule R5) for one passage vector
// ---------------------------------------------------------------------
// CALL db.index.vector.queryNodes('capec_emb', 10, $passage_vector) YIELD node AS c, score
// MATCH (c)-[:EXPLOITS_WEAKNESS]->(w:CWE)
// RETURN w.id AS cwe, sum(2 * score - 1) AS bridge_score
// ORDER BY bridge_score DESC;
