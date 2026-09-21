"""Connexion PostgreSQL/pgvector pour la persistance des embeddings.

matching-api reste utilisable sans base (fallback FAISS en mémoire, voir
main.py) : cette couche est optionnelle et ne doit jamais faire planter le
service si POSTGRES_DSN est absent ou injoignable.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_dsn() -> str:
    """DSN Postgres, forcé sur le driver psycopg (v3) installé dans l'image.

    POSTGRES_DSN est construit ailleurs (docker-compose) au format générique
    "postgresql://...", compatible psycopg2 par défaut chez SQLAlchemy ; on
    force ici "+psycopg" plutôt que d'ajouter psycopg2-binary en double.
    """
    dsn = os.getenv("POSTGRES_DSN", "")
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return dsn


def get_engine() -> Optional[Engine]:
    global _engine, _SessionLocal

    dsn = get_dsn()
    if not dsn:
        return None

    if _engine is None:
        _engine = create_engine(dsn, pool_pre_ping=True, pool_size=5, max_overflow=5)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    return _engine


def init_db() -> bool:
    """Prépare l'extension pgvector et les tables si la base est configurée.

    Retourne True si la persistance des embeddings est disponible, False
    sinon (auquel cas main.py retombe sur le calcul FAISS en mémoire).
    """
    engine = get_engine()
    if engine is None:
        logging.info("POSTGRES_DSN absent, embeddings persistants désactivés")
        return False

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            Base.metadata.create_all(bind=conn)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning("Initialisation pgvector impossible, fallback FAISS: %s", exc)
        return False


@contextmanager
def session_scope() -> Iterator[Session]:
    if _SessionLocal is None:
        raise RuntimeError("La base de données n'est pas initialisée")

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
