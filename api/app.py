# -*- coding: utf-8 -*-
import logging
from fastapi import FastAPI
from api.middleware.logging import LoggingMiddleware
from api.routes.health import router as health_router
from api.routes.vk_callback import router as vk_router
from config import ensure_dirs

logger = logging.getLogger("koib.api")

app = FastAPI(
    title="KOIB RAG API",
    description="Оптимизированная RAG-система для технической документации",
    version="4.6",
)

app.add_middleware(LoggingMiddleware)
app.include_router(health_router, tags=["health"])
app.include_router(vk_router, tags=["vk"])

@app.on_event("startup")
async def startup_event():
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("KOIB RAG API v4.6 запущен")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке сервера."""
    try:
        from api.routes.vk_callback import _generator
        if _generator and _generator.llm._session and not _generator.llm._session.closed:
            await _generator.llm._session.close()
            logger.info("aiohttp сессия закрыта")
    except Exception as e:
        logger.debug(f"Ошибка при закрытии сессии: {e}")
    logger.info("KOIB RAG API v4.6 остановлен")
