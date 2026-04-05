import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import config
from backend.app_state import sessions
from backend.routes.chat import router as chat_router
from backend.routes.health import router as health_router
from backend.routes.pipeline import router as pipeline_router
from backend.routes.sessions import router as sessions_router
from backend.session_store import load_all_sessions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        config.validate()
    except RuntimeError as e:
        logger.critical("Configuration error — server cannot start: %s", e)
        raise

    try:
        import nat.nat_config  # noqa: F401
        logger.info("NAT config validated OK")
    except RuntimeError as e:
        logger.warning("NAT config missing — Node 2/3 will fail at runtime: %s", e)

    logger.info("Server starting — loading sessions from GCS")
    sessions.update(await load_all_sessions())
    logger.info("Server ready | sessions_loaded=%d", len(sessions))
    yield
    logger.info("Server shutting down")


app = FastAPI(title="ArXiv Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception | path=%s | method=%s | error=%s",
        request.url.path,
        request.method,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(pipeline_router)
app.include_router(chat_router)
