"""Stay Assistant: LangGraph agent that finds stays via MongoDB Vector Search."""
import uuid

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import create_react_agent

from src.config import CHAT_MODEL, CHECKPOINT_DB
from src.db import client
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are Stay Assistant, an assistant that helps users find Airbnb \
stays that match what they are looking for.
- Use the `search_stays` tool to search the database; never invent listings.
- If key information is missing (city, number of guests), ask before searching.
- Keep answers short, highlighting name, location, price per night, capacity, and why it matches.
- Remember the preferences the user mentioned during the conversation."""


def build_agent():
    checkpointer = MongoDBSaver(client, db_name=CHECKPOINT_DB)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return create_react_agent(
        llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def ask(agent, question: str, thread_id: str) -> str:
    result = agent.invoke(
        {"messages": [("user", question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    # Quick terminal test: python -m src.agent
    agent = build_agent()
    thread = str(uuid.uuid4())
    print("Stay Assistant (Ctrl+C to quit)")
    while True:
        q = input("\nYou: ")
        print("\nAssistant:", ask(agent, q, thread))
