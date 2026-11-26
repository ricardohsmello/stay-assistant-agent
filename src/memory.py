"""Long-term user preference memory.

Preferences are extracted in the background after each exchange (not by the agent
itself), then stored as short free-text phrases per user.
"""
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import EXTRACTION_MODEL
from src.db import preferences


class PreferenceUpdate(BaseModel):
    add: list[str] = Field(
        default_factory=list,
        description="New durable preferences, as short English phrases.",
    )
    remove: list[str] = Field(
        default_factory=list,
        description="Existing preferences (copied exactly) that the user has now contradicted.",
    )


EXTRACTION_PROMPT = """You maintain the long-term travel-stay preference memory of a user.

Existing preferences:
{existing}

Latest exchange:
User: {question}
Assistant: {answer}

Return preferences to add and existing ones to remove.
Rules:
- Only durable tastes and needs (e.g. "beach view", "quiet neighborhood", "needs a kitchen", \
"travels with a dog"). Phrase dislikes as such (e.g. "dislikes shared rooms").
- Ignore one-off trip details: dates, a specific city for this trip, a specific listing, budget for one trip.
- Only use what the USER said or clearly implied. Do not add what the assistant suggested.
- Do not repeat existing preferences. Remove an existing one only if the user now contradicts it.
- If there is nothing to change, return empty lists."""

_extractor = ChatOpenAI(model=EXTRACTION_MODEL, temperature=0).with_structured_output(
    PreferenceUpdate
)


def get_user_preferences(user_id: str) -> list[str]:
    doc = preferences.find_one({"user_id": user_id}) or {}
    return doc.get("preferences", [])


def extract_and_save(user_id: str, question: str, answer: str) -> None:
    """Extract preferences from the latest exchange and update the user's memory."""
    try:
        existing = get_user_preferences(user_id)
        update: PreferenceUpdate = _extractor.invoke(
            EXTRACTION_PROMPT.format(
                existing=existing or "(none)", question=question, answer=answer
            )
        )
        if not update.add and not update.remove:
            return

        now = datetime.now(timezone.utc)
        if update.remove:
            preferences.update_one(
                {"user_id": user_id}, {"$pullAll": {"preferences": update.remove}}
            )
        if update.add:
            preferences.update_one(
                {"user_id": user_id},
                {"$addToSet": {"preferences": {"$each": update.add}}, "$set": {"updated_at": now}},
                upsert=True,
            )
        print(f"\n[memory] user={user_id} add={update.add} remove={update.remove}")
    except Exception as e:  # never break the chat because of memory
        print(f"\n[memory] extraction failed: {e}")
