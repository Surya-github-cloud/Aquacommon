# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Aquacommons Environment.

This module creates an HTTP server that exposes the AquacommonsEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import AquacommonsAction, AquacommonsObservation
    from .environment import AquacommonsEnvironment
except ImportError:
    from models import AquacommonsAction, AquacommonsObservation
    from server.environment import AquacommonsEnvironment


# Create the app with web interface and README integration
app = create_app(
    AquacommonsEnvironment,
    AquacommonsAction,
    AquacommonsObservation,
    env_name="aquacommons",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# HF Spaces healthcheck endpoints
@app.get("/")
async def root():
    """Root endpoint for HF Spaces healthcheck."""
    return {
        "status": "ok",
        "environment": "aquacommons",
        "version": "0.1.0",
        "message": "AquaCommons sustainable fishing OpenEnv ready"
    }

@app.get("/health")
async def health():
    """Health check endpoint for deployments."""
    return {
        "status": "healthy",
        "service": "aquacommons-env",
        "endpoints": ["/reset", "/step", "/state", "/schema"]
    }


@app.on_event("startup")
async def startup_message():
    """Print user-friendly startup message."""
    print("\n" + "="*70)
    print("🐟 AquaCommons OpenEnv Server Started")
    print("="*70)
    print(" Local:      http://localhost:8000")
    print(" API Schema: http://localhost:8000/schema")
    print(" Health:     http://localhost:8000/health")
    print("\n💡 Use http://localhost:8000 ")
    print("="*70 + "\n")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m aquacommons.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn aquacommons.server.app:app --workers 4
    """
    import sys
    import argparse

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--host", type=str, default="0.0.0.0")
        args = parser.parse_args()
        host = args.host
        port = args.port

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
