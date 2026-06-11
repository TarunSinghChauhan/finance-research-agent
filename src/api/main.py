from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.core.database import create_tables
from src.core.logging import setup_logging, get_logger
from src.api.routers import research, health

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", service="finance-research-agent")
    await create_tables()
    yield
    logger.info("shutdown", service="finance-research-agent")


app = FastAPI(
    title="Autonomous Financial Research Agent",
    description="AI agent that researches companies using structured 4-step reasoning chains with full audit trails",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(research.router, prefix="/research", tags=["Research"])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    with open("/app/src/api/dashboard.html") as f:
        return f.read()
