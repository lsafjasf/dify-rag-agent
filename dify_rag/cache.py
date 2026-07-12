"""
LLM 响应缓存 — SQLite 持久化, 避免 HyDE / QE 重复调用 LLM。
============================================================
同一个问题 + 同一个 LLM 模型 → 直接从 SQLite 读取缓存结果。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional


class QueryCache:
    """基于 SQLite 的持久化查询缓存。

    用法:
      cache = QueryCache("./cache")
      cache.set("hyde", "如何创建知识库?", "生成的技术文档...")
      result = cache.get("hyde", "如何创建知识库?")
    """

    def __init__(self, cache_dir: str = "./cache"):
        self._db_path = Path(cache_dir) / "llm_cache.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.commit()

    @staticmethod
    def _make_key(prefix: str, question: str) -> str:
        """生成缓存 key: prefix + md5(question)。"""
        q_hash = hashlib.md5(question.encode("utf-8")).hexdigest()
        return f"{prefix}:{q_hash}"

    def get(self, prefix: str, question: str) -> Optional[str]:
        """读取缓存, 命中时更新 access_count。"""
        key = self._make_key(prefix, question)
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    "SELECT value FROM query_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE query_cache SET access_count = access_count + 1 WHERE cache_key = ?",
                        (key,),
                    )
                    conn.commit()
                    return row[0]
        except Exception:
            pass
        return None

    def set(self, prefix: str, question: str, value: str) -> None:
        """写入缓存。"""
        key = self._make_key(prefix, question)
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO query_cache (cache_key, value, created_at, access_count) VALUES (?, ?, ?, 1)",
                    (key, value, time.time()),
                )
                conn.commit()
        except Exception:
            pass

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                total = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
                hits = conn.execute(
                    "SELECT SUM(access_count) FROM query_cache"
                ).fetchone()[0] or 0
                return {"total_entries": total, "total_hits": hits, "db_path": str(self._db_path)}
        except Exception:
            return {"total_entries": 0, "total_hits": 0, "db_path": str(self._db_path)}

    def clear(self) -> None:
        """清空所有缓存。"""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("DELETE FROM query_cache")
                conn.commit()
        except Exception:
            pass


# 全局单例
_cache_instance: Optional["QueryCache"] = None
_embedding_cache_instance: Optional["EmbeddingCache"] = None


def get_cache(cache_dir: str = "./cache") -> "QueryCache":
    """获取全局 LLM 缓存单例。"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache(cache_dir)
    return _cache_instance


# --------------------------------------------------------
# Embedding 缓存 — 按模型隔离
# --------------------------------------------------------


class EmbeddingCache:
    """基于 SQLite 的 embedding 向量缓存。

    缓存 key = md5(model_name + question)
    不同模型的 embedding 互相隔离，切换模型自动使用各自缓存。

    用法:
      ecache = EmbeddingCache("./cache")
      vec = ecache.get("text-embedding-v1", "如何创建知识库?")
      if vec is None:
          vec = embedding_api.call("如何创建知识库?")
          ecache.set("text-embedding-v1", "如何创建知识库?", vec)
    """

    def __init__(self, cache_dir: str = "./cache"):
        self._db_path = Path(cache_dir) / "embedding_cache.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    question TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_model
                ON embedding_cache(model)
            """)
            conn.commit()

    @staticmethod
    def _make_key(model: str, text: str) -> str:
        """生成缓存 key: md5(model + text)。"""
        raw = f"{model}||{text}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> Optional[List[float]]:
        """读取缓存的 embedding 向量。

        Args:
            model: embedding 模型名称。
            text: 原始文本。

        Returns:
            embedding 向量列表, 缓存未命中返回 None。
        """
        key = self._make_key(model, text)
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    "SELECT embedding FROM embedding_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE embedding_cache SET access_count = access_count + 1 WHERE cache_key = ?",
                        (key,),
                    )
                    conn.commit()
                    return json.loads(row[0])
        except Exception:
            pass
        return None

    def set(self, model: str, text: str, embedding: List[float]) -> None:
        """写入 embedding 缓存。

        Args:
            model: embedding 模型名称。
            text: 原始文本。
            embedding: embedding 向量。
        """
        key = self._make_key(model, text)
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embedding_cache "
                    "(cache_key, model, question, embedding, dimension, created_at, access_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (
                        key,
                        model,
                        text,
                        json.dumps(embedding),
                        len(embedding),
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception:
            pass

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM embedding_cache"
                ).fetchone()[0]
                hits = conn.execute(
                    "SELECT SUM(access_count) FROM embedding_cache"
                ).fetchone()[0] or 0
                models = conn.execute(
                    "SELECT model, COUNT(*) as cnt FROM embedding_cache GROUP BY model"
                ).fetchall()
                return {
                    "total_entries": total,
                    "total_hits": hits,
                    "models": {m: c for m, c in models},
                    "db_path": str(self._db_path),
                }
        except Exception:
            return {"total_entries": 0, "total_hits": 0, "models": {}, "db_path": str(self._db_path)}

    def clear(self, model: Optional[str] = None) -> None:
        """清空缓存。

        Args:
            model: 指定模型名则只清该模型的缓存, None 则全部清空。
        """
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                if model:
                    conn.execute(
                        "DELETE FROM embedding_cache WHERE model = ?", (model,)
                    )
                else:
                    conn.execute("DELETE FROM embedding_cache")
                conn.commit()
        except Exception:
            pass


def get_embedding_cache(cache_dir: str = "./cache") -> "EmbeddingCache":
    """获取全局 embedding 缓存单例。"""
    global _embedding_cache_instance
    if _embedding_cache_instance is None:
        _embedding_cache_instance = EmbeddingCache(cache_dir)
    return _embedding_cache_instance
