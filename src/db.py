"""MongoDB connection."""
from pymongo import MongoClient

from src.config import (
    CHECKPOINT_DB,
    COLLECTION,
    DB_NAME,
    MONGODB_URI,
    PREFERENCES_COLLECTION,
    THREADS_COLLECTION,
    USERS_COLLECTION,
)

client = MongoClient(MONGODB_URI)
listings = client[DB_NAME][COLLECTION]

_app_db = client[CHECKPOINT_DB]
preferences = _app_db[PREFERENCES_COLLECTION]
users = _app_db[USERS_COLLECTION]
threads = _app_db[THREADS_COLLECTION]
