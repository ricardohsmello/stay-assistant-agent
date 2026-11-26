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

CHECKPOINT_DB = "stay_assistant"
PREFERENCES_COLLECTION = "user_preferences"
USERS_COLLECTION = "users"
THREADS_COLLECTION = "threads"
EXTRACTION_MODEL = "gpt-4o-mini"  # cheap model for background preference extraction

# No login yet: a single default user, optionally overridden via .env (STAY_USER_ID)
USER_ID = os.getenv("STAY_USER_ID", "default_user")
