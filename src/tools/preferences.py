"""Tools for reading long-term user preferences."""
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.config import USER_ID
from src.memory import get_user_preferences


@tool
def get_preferences(config: RunnableConfig) -> list[str]:
    """Get the user's saved long-term preferences (empty list if none)."""
    user_id = (config.get("configurable") or {}).get("user_id", USER_ID)
    prefs = get_user_preferences(user_id)
    print(f"\n[get_preferences] {prefs}")
    return prefs
