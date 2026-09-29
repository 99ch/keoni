"""bookkeeping table for the autonomous scoring loop

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-29

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "job_scoring_state",
        sa.Column("job_id", sa.Integer(), primary_key=True),
        sa.Column("last_scored_job_modified", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("last_scored_cv_pool_version", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "last_scored_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("job_scoring_state")
