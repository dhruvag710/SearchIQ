"""Add document_images table for visual PDF assets.

Revision ID: 002
Revises: 001
Create Date: 2026-09-12

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, Sequence[str], None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create document_images table."""
    op.create_table(
        "document_images",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("image_index", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.String(length=1024), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_document_images_document_id"),
        "document_images",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop document_images table."""
    op.drop_index(op.f("ix_document_images_document_id"), table_name="document_images")
    op.drop_table("document_images")
