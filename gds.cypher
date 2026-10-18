// Graph Data Science over the corpus_kb graph (Neo4j GDS 2.13).
//
// Only the knowledge-base layer is projected: ATT&CK techniques and tactics,
// CWE, CAPEC and CVE with the six corpus_kb relationship types. The report
// layer (ThreatReport, ReportChunk, ReportBehavior) is left out, so no score
// carries information from the reports the questions are drawn from.
//
// Run all of it with cypher-shell, or step by step from scripts/gds_analysis.py,
// which reads the `// name:` blocks below. Results are written back as node
// properties: degree, component, pagerank, community, betweenness.
//
//     cat gds.cypher | docker exec -i cti-neo4j cypher-shell -u neo4j -p <password>

// name: drop
CALL gds.graph.drop('kb', false) YIELD graphName RETURN graphName;

// name: drop_directed
CALL gds.graph.drop('kb_directed', false) YIELD graphName RETURN graphName;

// Undirected view for structure: components, communities, bridges.
// name: project
CALL gds.graph.project('kb',
  ['AttackTechnique', 'Tactic', 'CWE', 'CAPEC', 'CVE'],
  {SUBTECHNIQUE_OF:   {orientation: 'UNDIRECTED'},
   REPLACED_BY:       {orientation: 'UNDIRECTED'},
   IN_TACTIC:         {orientation: 'UNDIRECTED'},
   EXPLOITS_WEAKNESS: {orientation: 'UNDIRECTED'},
   MAPS_TO_TECHNIQUE: {orientation: 'UNDIRECTED'},
   HAS_WEAKNESS:      {orientation: 'UNDIRECTED'}})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;

// Stored direction for PageRank: sub-technique -> parent, technique -> tactic,
// CAPEC -> CWE, CAPEC -> technique, CVE -> CWE, retired -> replacement. A node
// ranks high when much of the knowledge base points at it.
// name: project_directed
CALL gds.graph.project('kb_directed',
  ['AttackTechnique', 'Tactic', 'CWE', 'CAPEC', 'CVE'],
  ['SUBTECHNIQUE_OF', 'REPLACED_BY', 'IN_TACTIC', 'EXPLOITS_WEAKNESS', 'MAPS_TO_TECHNIQUE', 'HAS_WEAKNESS'])
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;

// name: degree
CALL gds.degree.write('kb', {writeProperty: 'degree'})
YIELD nodePropertiesWritten, centralityDistribution
RETURN nodePropertiesWritten, centralityDistribution;

// name: wcc
CALL gds.wcc.write('kb', {writeProperty: 'component'})
YIELD componentCount, componentDistribution
RETURN componentCount, componentDistribution;

// name: pagerank
CALL gds.pageRank.write('kb_directed', {writeProperty: 'pagerank', dampingFactor: 0.85, maxIterations: 50})
YIELD ranIterations, didConverge, centralityDistribution
RETURN ranIterations, didConverge, centralityDistribution;

// Single-threaded so the community IDs are reproducible.
// name: louvain
CALL gds.louvain.write('kb', {writeProperty: 'community', concurrency: 1})
YIELD communityCount, modularity, modularities, ranLevels, communityDistribution
RETURN communityCount, modularity, modularities, ranLevels, communityDistribution;

// name: betweenness
CALL gds.betweenness.write('kb', {writeProperty: 'betweenness'})
YIELD centralityDistribution
RETURN centralityDistribution;

// Weakest (shortest) evidence path between two nodes, e.g. a CVE and a technique.
// name: path
MATCH (a {id: $source}), (b {id: $target})
CALL gds.shortestPath.dijkstra.stream('kb', {sourceNode: a, targetNode: b})
YIELD totalCost, path
RETURN totalCost AS hops, [n IN nodes(path) | coalesce(n.id, n.name)] AS nodes;
