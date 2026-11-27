"""Agent tools. They are registered on the platform in src/main.py."""
from src.tools.booking import book_stay, check_availability
from src.tools.search import search_stays

ALL_TOOLS = [search_stays, check_availability, book_stay]
