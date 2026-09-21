from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Rend l'import de `app` fiable quel que soit le répertoire courant depuis
# lequel `alembic` est invoqué (alembic.ini::prepend_sys_path = . ne
# fonctionne que si le CWD est déjà le bon) — même correctif qu'AI Real-Time
# (4e1b6df), qui l'a ajouté après un import cassé en pratique.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import get_dsn  # noqa: E402
from app.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

dsn = get_dsn()
if dsn:
    config.set_main_option("sqlalchemy.url", dsn)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
