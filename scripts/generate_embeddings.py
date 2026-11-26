"""Generate OpenAI embeddings for sample_airbnb.listingsAndReviews and save them in the `embedding` field.

Usage:
    python scripts/generate_embeddings.py            # only docs without embedding
    python scripts/generate_embeddings.py --force    # regenerate all
    python scripts/generate_embeddings.py --limit 50 # test with a few docs
"""
import argparse
import os
import time

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient, UpdateOne

load_dotenv()

DB_NAME = "sample_airbnb"
COLLECTION = "listingsAndReviews"
MODEL = "text-embedding-3-small"  # 1536 dimensions
EMBEDDING_FIELD = "embedding"
BATCH_SIZE = 64

PROJECTION = {
    "name": 1,
    "description": 1,
    "neighborhood_overview": 1,
    "transit": 1,
    "property_type": 1,
    "room_type": 1,
    "amenities": 1,
    "address.market": 1,
    "address.suburb": 1,
    "address.country": 1,
}


def build_text(doc: dict) -> str:
    addr = doc.get("address") or {}
    parts = [
        f"Name: {doc.get('name', '')}",
        f"Property type: {doc.get('property_type', '')} ({doc.get('room_type', '')})",
        f"Location: {addr.get('suburb', '')}, {addr.get('market', '')}, {addr.get('country', '')}",
        f"Amenities: {', '.join(doc.get('amenities') or [])}",
        f"Description: {doc.get('description', '')}",
        f"Neighborhood: {doc.get('neighborhood_overview', '')}",
        f"Transit: {doc.get('transit', '')}",
    ]
    return "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())


def embed_with_retry(embedder, texts, retries=5):
    for attempt in range(retries):
        try:
            return embedder.embed_documents(texts)
        except Exception as e:  # rate limit / transient error
            wait = 2**attempt
            print(f"  error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Failed to generate embeddings after several attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    client = MongoClient(os.environ["MONGODB_URI"])
    col = client[DB_NAME][COLLECTION]
    embedder = OpenAIEmbeddings(model=MODEL)

    query = {} if args.force else {EMBEDDING_FIELD: {"$exists": False}}
    total = col.count_documents(query)
    if args.limit:
        total = min(total, args.limit)
    print(f"{total} documents to process")

    cursor = col.find(query, PROJECTION)
    if args.limit:
        cursor = cursor.limit(args.limit)

    done, batch = 0, []

    def flush(batch):
        vectors = embed_with_retry(embedder, [build_text(d) for d in batch])
        ops = [
            UpdateOne({"_id": d["_id"]}, {"$set": {EMBEDDING_FIELD: v}})
            for d, v in zip(batch, vectors)
        ]
        col.bulk_write(ops, ordered=False)

    for doc in cursor:
        batch.append(doc)
        if len(batch) >= BATCH_SIZE:
            flush(batch)
            done += len(batch)
            batch = []
            print(f"{done}/{total}")
    if batch:
        flush(batch)
        done += len(batch)
        print(f"{done}/{total}")

    print("Done.")


if __name__ == "__main__":
    main()
