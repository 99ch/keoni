"""job embeddings by chunk, not one vector per whole job

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-29

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.create_table(
        "job_embedding_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.UniqueConstraint("job_id", "chunk_index", name="ux_job_embedding_chunks_job_id_chunk_index"),
    )
    op.create_index("ix_job_embedding_chunks_job_id", "job_embedding_chunks", ["job_id"])
    op.execute(
        "CREATE INDEX ix_job_embedding_chunks_vector ON job_embedding_chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.execute("DROP INDEX IF EXISTS ix_job_embeddings_vector")
    op.drop_column("job_embeddings", "embedding")


def downgrade() -> None:
    op.add_column("job_embeddings", sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True))
    op.execute(
        "CREATE INDEX ix_job_embeddings_vector ON job_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.drop_table("job_embedding_chunks")
