"""HTTP API for the support assistant"""

from fastapi import FastAPI

app = FastAPI(title="support-assistant")


@app.get("/health")
def health():
    """Liveness check: the process is up an able to answer"""
    return {"status": "ok"}
