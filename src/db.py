"""MongoDB connection."""
from pymongo import MongoClient

from src.config import COLLECTION, DB_NAME, MONGODB_URI

client = MongoClient(MONGODB_URI)
listings = client[DB_NAME][COLLECTION]
