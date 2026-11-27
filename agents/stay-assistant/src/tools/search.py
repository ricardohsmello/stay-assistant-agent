"""Tools for searching stays."""
from typing import Optional

from langchain_openai import OpenAIEmbeddings

from src.config import EMBEDDING_FIELD, EMBEDDING_MODEL, VECTOR_INDEX
from src.db import listings

# check_embedding_ctx_length=False: skips downloading the tiktoken file (blocked by egress)
embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, check_embedding_ctx_length=False)


def search_stays(
    query: str,
    market: Optional[str] = None,
    room_type: Optional[str] = None,
    min_guests: Optional[int] = None,
    limit: int = 5,
) -> list[dict]:
    """Search stays by semantic similarity.

    Args:
        query: description of what the user wants (e.g. "quiet flat near the beach with kitchen").
        market: exact city/market, e.g. "Rio De Janeiro", "New York", "Barcelona". Only set it if the
            user explicitly named a city; otherwise leave it empty.
        room_type: "Entire home/apt", "Private room" or "Shared room".
        min_guests: minimum capacity (accommodates).
        limit: number of results.
    """
    pre_filter = {}
    if market:
        pre_filter["address.market"] = market
    if room_type:
        pre_filter["room_type"] = room_type
    if min_guests:
        pre_filter["accommodates"] = {"$gte": min_guests}

    vector_stage = {
        "index": VECTOR_INDEX,
        "path": EMBEDDING_FIELD,
        "queryVector": embedder.embed_query(query),
        "numCandidates": limit * 20,
        "limit": limit,
    }

    print( "vector search pronto pra executar!")

    if pre_filter:
        vector_stage["filter"] = pre_filter

    pipeline = [
        {"$vectorSearch": vector_stage},
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "summary": 1,
                "listing_url": 1,
                "property_type": 1,
                "room_type": 1,
                "accommodates": 1,
                "bedrooms": 1,
                "price": 1,
                "amenities": 1,
                "address.market": 1,
                "address.suburb": 1,
                "review_scores.review_scores_rating": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = []
    for doc in listings.aggregate(pipeline):
        if doc.get("price") is not None:
            doc["price"] = float(doc["price"].to_decimal())
        doc["amenities"] = (doc.get("amenities") or [])[:10]
        results.append(doc)

    print(f"\n[search_stays] query={query!r} filter={pre_filter or None}")
    for r in results:
        print(f"  - {r['name']}")
    # A made-up or too narrow city returns nothing: retry once without the city filter.
    if not results and market:
        print("  (no results, retrying without the market filter)")
        return search_stays(query, None, room_type, min_guests, limit)
    return results
