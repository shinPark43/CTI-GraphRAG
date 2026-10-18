"""FastAPI service over ctiattr.service (design document, Section 8).

    uv run uvicorn ctiattr.api:app --port 8000
    curl -s localhost:8000/attribute -H 'content-type: application/json' \
         -d '{"passage": "...", "taxonomy": "attack"}'
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ctiattr.schemas import AttributionRequest, AttributionResponse, PathStep
from ctiattr.service import AttributionService, dev_examples

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph = None
    try:
        from ctiattr.graph_db import GraphDB
        graph = GraphDB()
        graph.run("RETURN 1")
    except Exception:  # noqa: BLE001 - answer without the graph analytics
        graph = None
    state["svc"] = AttributionService(graph=graph)
    yield
    if graph is not None:
        graph.close()


app = FastAPI(title="CTI attribution (SGS)", lifespan=lifespan)


class Verify(BaseModel):
    path: list[PathStep]


@app.get("/health")
def health() -> dict:
    svc = state["svc"]
    return {"model": svc.model, "neo4j": svc.graph is not None}


@app.post("/attribute", response_model=AttributionResponse)
def attribute(req: AttributionRequest) -> AttributionResponse:
    return state["svc"].attribute(req)


@app.get("/graph/{node_id}")
def graph_context(node_id: str) -> dict:
    ctx = state["svc"].graph_context(node_id)
    if ctx is None:
        raise HTTPException(404, "node not found or Neo4j unavailable")
    return ctx


@app.post("/verify")
def verify(v: Verify) -> dict:
    try:
        return {"paths": state["svc"].verify(v.path)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/examples")
def examples() -> list[dict]:
    return [{"id": qa.id, "task": qa.task, "passage": qa.passage, "report_id": qa.report_id, "gold": qa.gold}
            for qa in dev_examples()]
