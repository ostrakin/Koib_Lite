# -*- coding: utf-8 -*-
import time
import logging
from typing import Dict, Any
from fastapi import APIRouter

logger = logging.getLogger("koib.api.health")
router = APIRouter()
_start_time = time.time()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    uptime = time.time() - _start_time
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "version": "4.6",
    }

@router.get("/")
async def root() -> Dict[str, str]:
    return {
        "name": "KOIB RAG API",
        "version": "4.6",
        "description": "Оптимизированная RAG-система для технической документации",
        "docs": "/docs",
    }
