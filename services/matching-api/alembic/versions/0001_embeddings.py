"""create pgvector embeddings tables

Revision ID: 0001
Revises:
Create Date: 2026-09-10

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "job_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("job_id", name="ux_job_embeddings_job_id"),
    )
    op.create_index("ix_job_embeddings_job_id", "job_embeddings", ["job_id"])
    op.execute(
        "CREATE INDEX ix_job_embeddings_vector ON job_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.create_table(
        "cv_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("cv_id", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("cv_id", name="ux_cv_embeddings_cv_id"),
    )
    op.create_index("ix_cv_embeddings_cv_id", "cv_embeddings", ["cv_id"])
    op.execute(
        "CREATE INDEX ix_cv_embeddings_vector ON cv_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.drop_table("cv_embeddings")
    op.drop_table("job_embeddings")
