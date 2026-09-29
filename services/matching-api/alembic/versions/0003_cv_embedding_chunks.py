"""cv embeddings by chunk, not one vector per whole cv

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-29

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.create_table(
        "cv_embedding_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("cv_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.UniqueConstraint("cv_id", "chunk_index", name="ux_cv_embedding_chunks_cv_id_chunk_index"),
    )
    op.create_index("ix_cv_embedding_chunks_cv_id", "cv_embedding_chunks", ["cv_id"])
    op.execute(
        "CREATE INDEX ix_cv_embedding_chunks_vector ON cv_embedding_chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.execute("DROP INDEX IF EXISTS ix_cv_embeddings_vector")
    op.drop_column("cv_embeddings", "embedding")


def downgrade() -> None:
    op.add_column("cv_embeddings", sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True))
    op.execute(
        "CREATE INDEX ix_cv_embeddings_vector ON cv_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.drop_table("cv_embedding_chunks")
