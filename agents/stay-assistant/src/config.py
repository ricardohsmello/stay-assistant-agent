"""Central configuration loaded from environment variables and constants."""
import os

from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]

DB_NAME = "sample_airbnb"
COLLECTION = "listingsAndReviews"
VECTOR_INDEX = "vector_index"
EMBEDDING_FIELD = "embedding"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
PLAN_MODEL = "gpt-4o-mini"  # cheap model for the short plan

APP_DB = "stay_assistant"  # where our own data lives (bookings)
