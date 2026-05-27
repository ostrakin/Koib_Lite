# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — Модуль индексации
★ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: BM25 на SQLite FTS5 (zero-RAM sparse search)
  Вместо загрузки корпуса в rank_bm25 (~300-500 МБ RAM на 1000+ страниц)
  используем SQLite FTS5, который работает с диска без загрузки в память.
★ DocStore выгружен в SQLite (full_content таблиц/формул).
★ Синглтон эмбеддингов с ленивой инициализацией.
"""
import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

import numpy as np

from config import (
    INDEX_DIR, DOCSTORE_DIR, METADATA_DIR,
    EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    OPENAI_API_KEY, BM25_USE_STOPWORDS, PASSAGE_PREFIX,
)

logger = logging.getLogger("koib.indexing")

# ═══════════════════════════════════════════════════════════════
# Синглтон эмбеддингов (ленивая загрузка, экономия RAM)
# ═══════════════════════════════════════════════════════════════
_GLOBAL_EMBEDDINGS = None


def get_global_embeddings():
    """Ленивый синглтон модели эмбеддингов."""
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is not None:
        return _GLOBAL_EMBEDDINGS

    if EMBEDDING_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        _GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        _GLOBAL_EMBEDDINGS = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

    logger.info(f"Эмбеддинги загружены: {EMBEDDING_PROVIDER}")
    return _GLOBAL_EMBEDDINGS


# ═══════════════════════════════════════════════════════════════
# Русская токенизация для FTS5
# ═══════════════════════════════════════════════════════════════
RU_STOPWORDS = {
    "и", "в", "на", "с", "по", "для", "из", "к", "от", "о", "об", "а", "но",
    "да", "не", "что", "как", "это", "то", "же", "бы", "вы", "мы", "он", "она",
    "они", "оно", "я", "ты", "его", "её", "их", "мой", "твой", "наш", "ваш",
    "свой", "этот", "тот", "такой", "который", "весь", "все", "вся", "всё",
    "быть", "был", "была", "было", "были", "будет", "есть", "нет", "ещё", "уже",
    "только", "если", "или", "при", "про", "за", "до", "после", "между",
    "через", "над", "под", "перед", "так", "тоже", "лишь", "ведь", "вот",
    "даже", "ну", "ли", "ни", "тебя", "мне", "мной", "ним", "ней", "нами",
    "вам", "вас", "нас", "них", "чего", "чему", "чем", "кем", "ком", "где",
    "когда", "зачем", "почему", "куда", "откуда", "какой", "какая", "какие",
}

_TOKEN_RE = re.compile(r'[а-яёa-z0-9]+', re.IGNORECASE)


def tokenize_ru(text: str) -> str:
    """
    Токенизация русского текста для FTS5.
    Возвращает строку токенов через пробел.
    """
    if not text:
        return ""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]
    if BM25_USE_STOPWORDS:
        tokens = [t for t in tokens if t not in RU_STOPWORDS]
    return " ".join(tokens)


def prepare_fts_query(query: str) -> str:
    """
    Готовит пользовательский запрос для FTS5 MATCH.
    Токенизирует и соединяет через OR (широкий поиск).
    Экранирует спецсимволы FTS5.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(query) if len(t) > 1]
    if BM25_USE_STOPWORDS:
        tokens = [t for t in tokens if t not in RU_STOPWORDS]
    if not tokens:
        return ""
    # OR-объединение для широкого recall
    return " OR ".join(f'"{t}"' for t in tokens[:20])


# ═══════════════════════════════════════════════════════════════
# DocStore: SQLite хранилище full_content (таблицы/формулы/рисунки)
# ═══════════════════════════════════════════════════════════════
class DocStore:
    """SQLite-хранилище полных версий структурированных чанков."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DOCSTORE_DIR / "docstore.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS docstore (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT,
                    chunk_type TEXT,
                    metadata TEXT
                )
            """)

    def add(self, chunk) -> None:
        if not chunk.full_content:
            return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO docstore "
                    "(chunk_id, content, chunk_type, metadata) VALUES (?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.full_content,
                        chunk.chunk_type,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    ),
                )
        except Exception as exc:
            logger.debug(f"DocStore add error: {exc}")

    def add_many(self, chunks) -> None:
        rows = [
            (c.chunk_id, c.full_content, c.chunk_type,
             json.dumps(c.metadata, ensure_ascii=False))
            for c in chunks if c.full_content
        ]
        if not rows:
            return
        try:
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO docstore "
                    "(chunk_id, content, chunk_type, metadata) VALUES (?, ?, ?, ?)",
                    rows,
                )
        except Exception as exc:
            logger.warning(f"DocStore add_many error: {exc}")

    def get_content(self, chunk_id: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM docstore WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None


# ═══════════════════════════════════════════════════════════════
# BM25 через SQLite FTS5 (zero-RAM sparse search)
# ═══════════════════════════════════════════════════════════════
class BM25FTSIndex:
    """
    Sparse-индекс на SQLite FTS5.
    Полностью заменяет rank_bm25: корпус НЕ загружается в RAM,
    поиск идёт с диска через встроенный BM25-ранжировщик FTS5.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (INDEX_DIR / "bm25_fts.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    chunk_type UNINDEXED,
                    source UNINDEXED,
                    page UNINDEXED,
                    heading UNINDEXED,
                    model UNINDEXED,
                    metadata UNINDEXED,
                    tokenize='unicode61 remove_diacritics 1'
                )
            """)

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks_fts")

    def add_chunks(self, chunks) -> None:
        """Добавить чанки в FTS-индекс (батчем для скорости)."""
        rows = []
        for c in chunks:
            # Таблицы/формулы индексируем по full_content, если есть
            text_for_index = c.full_content if c.full_content else c.content
            tokenized = tokenize_ru(text_for_index)
            if not tokenized:
                continue
            rows.append((
                c.chunk_id,
                tokenized,
                c.chunk_type,
                c.metadata.get("source", ""),
                str(c.metadata.get("page", 0)),
                c.metadata.get("heading", ""),
                c.metadata.get("model", "unknown"),
                json.dumps(c.metadata, ensure_ascii=False),
            ))
        if not rows:
            return
        try:
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO chunks_fts "
                    "(chunk_id, content, chunk_type, source, page, heading, model, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            logger.info(f"FTS5: добавлено {len(rows)} чанков")
        except Exception as exc:
            logger.warning(f"FTS5 add_chunks error: {exc}")

    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Возвращает список (metadata_dict, score).
        FTS5 bm25() возвращает отрицательные числа — инвертируем для единообразия.
        """
        fts_query = prepare_fts_query(query)
        if not fts_query:
            return []
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT chunk_id, content, chunk_type, source, page,
                       heading, model, metadata, bm25(chunks_fts) AS rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, k),
            )
            results = []
            for row in cur.fetchall():
                try:
                    metadata = json.loads(row[7]) if row[7] else {}
                except Exception:
                    metadata = {}
                # Нормализуем ключи (FTS хранит как строки)
                metadata.setdefault("chunk_id", row[0])
                metadata.setdefault("chunk_type", row[2])
                metadata.setdefault("source", row[3])
                metadata.setdefault("page", int(row[4]) if row[4] else 0)
                metadata.setdefault("heading", row[5])
                metadata.setdefault("model", row[6])
                metadata.setdefault("content", row[1])
                # bm25() < 0 → инвертируем, чтобы больше = лучше
                score = -float(row[8]) if row[8] is not None else 0.0
                results.append((metadata, score))
            return results
        except Exception as exc:
            logger.warning(f"FTS5 search error: {exc}")
            return []

    def count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks_fts")
        row = cur.fetchone()
        return row[0] if row else 0


# ═══════════════════════════════════════════════════════════════
# IndexBuilder: собирает FAISS + FTS5 + DocStore
# ═══════════════════════════════════════════════════════════════
class IndexBuilder:
    """
    Построитель поисковых индексов.
    - text_vectorstore:    FAISS для текстовых чанков
    - summary_vectorstore: FAISS для сводок таблиц/формул/рисунков
    - bm25:                SQLite FTS5 (sparse search, zero-RAM)
    - docstore:            SQLite DocStore (full_content)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else INDEX_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.text_vectorstore = None
        self.summary_vectorstore = None
        self.bm25 = BM25FTSIndex(self.output_dir / "bm25_fts.db")
        self.docstore = DocStore(DOCSTORE_DIR / "docstore.db")

        self._text_docs: List = []
        self._summary_docs: List = []

    def add_chunks(self, chunks) -> None:
        """Добавить чанки во все индексы (батч-режим)."""
        from langchain_core.documents import Document

        # 1. DocStore: сохраняем full_content для структурированных чанков
        self.docstore.add_many(chunks)

        # 2. FTS5: sparse-индекс
        self.bm25.add_chunks(chunks)

        # 3. Раскладываем по векторным базам
        for c in chunks:
            lc_doc = c.to_langchain_doc()
            if c.chunk_type == "text":
                self._text_docs.append(lc_doc)
            else:
                # Для таблиц/формул индексируем эвристическую сводку
                self._summary_docs.append(lc_doc)

        # Периодическая сборка мусора при больших батчах
        if len(self._text_docs) + len(self._summary_docs) > 2000:
            self._flush_vectorstores()

    def _flush_vectorstores(self) -> None:
        """Собрать FAISS-индексы из накопленных документов."""
        if not self._text_docs and not self._summary_docs:
            return

        embeddings = get_global_embeddings()

        try:
            from langchain_community.vectorstores import FAISS

            if self._text_docs:
                if self.text_vectorstore is None:
                    self.text_vectorstore = FAISS.from_documents(
                        self._text_docs, embeddings
                    )
                else:
                    self.text_vectorstore.add_documents(self._text_docs)
                self.text_vectorstore.save_local(
                    str(self.output_dir), index_name="text_index"
                )
                logger.info(f"FAISS text: {len(self._text_docs)} docs added")
                self._text_docs = []

            if self._summary_docs:
                if self.summary_vectorstore is None:
                    self.summary_vectorstore = FAISS.from_documents(
                        self._summary_docs, embeddings
                    )
                else:
                    self.summary_vectorstore.add_documents(self._summary_docs)
                self.summary_vectorstore.save_local(
                    str(self.output_dir), index_name="summary_index"
                )
                logger.info(f"FAISS summary: {len(self._summary_docs)} docs added")
                self._summary_docs = []
        except Exception as exc:
            logger.error(f"Ошибка сборки FAISS: {exc}")

    def save(self) -> None:
        """Финальная сборка и сохранение всех индексов."""
        self._flush_vectorstores()
        logger.info(
            f"Индексы сохранены. FTS5 чанков: {self.bm25.count()}"
        )

    def load(self) -> None:
        """Загрузить существующие FAISS-индексы с диска."""
        embeddings = get_global_embeddings()
        try:
            from langchain_community.vectorstores import FAISS

            text_path = self.output_dir / "text_index.faiss"
            if text_path.exists():
                self.text_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings,
                    index_name="text_index", allow_dangerous_deserialization=True,
                )
                logger.info("FAISS text_index загружен")

            summary_path = self.output_dir / "summary_index.faiss"
            if summary_path.exists():
                self.summary_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings,
                    index_name="summary_index", allow_dangerous_deserialization=True,
                )
                logger.info("FAISS summary_index загружен")
        except Exception as exc:
            logger.warning(f"Ошибка загрузки FAISS: {exc}")

        logger.info(f"FTS5 чанков в индексе: {self.bm25.count()}")