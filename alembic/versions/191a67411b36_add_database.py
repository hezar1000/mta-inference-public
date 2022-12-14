"""Add database

Revision ID: 191a67411b36
Revises: 
Create Date: 2022-02-26 13:20:32.362679

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '191a67411b36'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('experiment',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('time_created', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('time_updated', sa.DateTime(timezone=True), nullable=True),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('dataset',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('time_created', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('time_updated', sa.DateTime(timezone=True), nullable=True),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('data', sa.LargeBinary(length=4294967295), nullable=False),
    sa.Column('held_out_data', sa.LargeBinary(length=4294967295), nullable=True),
    sa.Column('settings', sa.JSON(), nullable=True),
    sa.Column('experiment_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiment.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('model_run',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('time_created', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('time_updated', sa.DateTime(timezone=True), nullable=True),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('settings', sa.JSON(), nullable=True),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('experiment_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['dataset.id'], ),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiment.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('samples',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('time_created', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('time_updated', sa.DateTime(timezone=True), nullable=True),
    sa.Column('model_run_id', sa.Integer(), nullable=False),
    sa.Column('data', sa.LargeBinary(length=4294967295), nullable=True),
    sa.Column('settings', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['model_run_id'], ['model_run.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('summary',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('time_created', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('time_updated', sa.DateTime(timezone=True), nullable=True),
    sa.Column('model_run_id', sa.Integer(), nullable=False),
    sa.Column('data', sa.LargeBinary(length=4294967295), nullable=True),
    sa.ForeignKeyConstraint(['model_run_id'], ['model_run.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('summary')
    op.drop_table('samples')
    op.drop_table('model_run')
    op.drop_table('dataset')
    op.drop_table('experiment')
    # ### end Alembic commands ###
