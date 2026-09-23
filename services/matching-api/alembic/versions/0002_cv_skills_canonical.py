"""add skills_canonical to cv_embeddings

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-23

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "cv_embeddings",
        sa.Column(
            "skills_canonical",
            ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.execute(
        "CREATE INDEX ix_cv_embeddings_skills_canonical ON cv_embeddings USING gin (skills_canonical)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_cv_embeddings_skills_canonical")
    op.drop_column("cv_embeddings", "skills_canonical")
