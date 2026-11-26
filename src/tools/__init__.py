"""Agent tools. Add new tools here to expose them to the agent."""
from src.tools.booking import book_stay, check_availability
from src.tools.search import search_stays

ALL_TOOLS = [search_stays, check_availability, book_stay]
