"""Users and their conversation threads (metadata only; messages live in checkpoints)."""
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.db import threads, users

DEFAULT_TITLE = "New chat"


def normalize_user(name: str) -> str:
    return name.strip().lower()


def list_users() -> list[str]:
    return sorted(u["user_id"] for u in users.find({}, {"user_id": 1}))


def create_user(user_id: str) -> None:
    users.update_one(
        {"user_id": user_id},
        {"$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


def list_threads(user_id: str) -> list[dict]:
    return list(threads.find({"user_id": user_id}).sort("updated_at", -1))


def get_thread(user_id: str, thread_id: str) -> Optional[dict]:
    return threads.find_one({"user_id": user_id, "thread_id": thread_id})


def create_thread(user_id: str) -> str:
    """Create a new thread, reusing the user's latest one if it is still empty."""
    empty = threads.find_one({"user_id": user_id, "title": DEFAULT_TITLE})
    if empty:
        return empty["thread_id"]
    thread_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    threads.insert_one(
        {
            "thread_id": thread_id,
            "user_id": user_id,
            "title": DEFAULT_TITLE,
            "created_at": now,
            "updated_at": now,
        }
    )
    return thread_id


def touch_thread(thread_id: str, first_message: str) -> None:
    """Bump updated_at; the first message becomes the title (first ~40 chars)."""
    now = datetime.now(timezone.utc)
    title = first_message.strip().replace("\n", " ")[:40]
    threads.update_one(
        {"thread_id": thread_id, "title": DEFAULT_TITLE},
        {"$set": {"title": title}},
    )
    threads.update_one({"thread_id": thread_id}, {"$set": {"updated_at": now}})
