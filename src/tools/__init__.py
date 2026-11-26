"""Agent tools. Add new tools here to expose them to the agent."""
from src.tools.preferences import get_preferences
from src.tools.search import search_stays

ALL_TOOLS = [search_stays, get_preferences]
