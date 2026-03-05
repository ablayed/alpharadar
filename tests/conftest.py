from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest_asyncio

from src.storage.database import Database


@pytest_asyncio.fixture
async def db():
    test_dir = Path(".test_tmp")
    test_dir.mkdir(exist_ok=True)
    db_path = test_dir / f"{uuid4().hex}.db"
    test_db = Database(str(db_path))
    await test_db.initialize()
    yield test_db
    db_path.unlink(missing_ok=True)
