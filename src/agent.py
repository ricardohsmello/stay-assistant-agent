"""Stay Assistant: LangGraph agent that finds stays via MongoDB Vector Search."""
import threading
import uuid

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command

from src.config import CHECKPOINT_DB, USER_ID
from src.db import client
from src.memory import extract_and_save, get_user_preferences
from src.workflow import build_workflow

SYSTEM_PROMPT = """You are Stay Assistant, an assistant that helps users find Airbnb \
stays that match what they are looking for.
- Use the `search_stays` tool to search the database; never invent listings.
- Always search right away with whatever the user has said so far; never ask questions before searching.
- After showing results, optionally offer to refine (e.g. by city, number of guests, or room type).
- Keep answers short, highlighting name, location, price per night, capacity, and why it matches.
- Remember the preferences the user mentioned during the conversation.

Booking:
- To book, you need the stay and the check-in/check-out dates; ask for the dates if missing.
- Always call `check_availability` first, then ask the user to confirm.
- Call `book_stay` only after the user explicitly confirms.

Long-term memory:
- The user's known preferences (if any) are listed below. Use them to enrich your search queries \
and recommendations, especially when the user's request is vague.
- Preferences are saved automatically by the system; you do not need to save them."""


def build_prompt(state, config: RunnableConfig):
    """Dynamic system prompt: base prompt + this user's saved preferences."""
    user_id = (config.get("configurable") or {}).get("user_id", USER_ID)
    prefs = get_user_preferences(user_id)
    memory = ", ".join(prefs) if prefs else "none yet"
    system = f"{SYSTEM_PROMPT}\n\nKnown preferences of this user: {memory}"
    if state.get("plan"):
        system += f"\n\nPlan for the current request:\n{state['plan']}"
    print(f"\n[prompt] user={user_id} preferences={prefs}")
    return [SystemMessage(content=system)] + state["messages"]


def build_agent():
    checkpointer = MongoDBSaver(client, db_name=CHECKPOINT_DB)
    return build_workflow(build_prompt, checkpointer)


def ask(agent, question: str, thread_id: str, user_id: str = USER_ID) -> str:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    # If the graph is paused waiting for a booking confirmation, this message is the answer.
    waiting = agent.get_state(config).next
    user_input = Command(resume=question) if waiting else {"messages": [("user", question)]}

    result = agent.invoke(user_input, config=config)
    if "__interrupt__" in result:  # paused: show the confirmation question
        answer = result["__interrupt__"][0].value
    else:
        answer = result["messages"][-1].content

    # Extract long-term preferences in the background so the reply isn't delayed.
    threading.Thread(
        target=extract_and_save, args=(user_id, question, answer), daemon=True
    ).start()
    return answer


def get_history(agent, thread_id: str) -> list[dict]:
    """Return the visible chat history (user/assistant text only) of a thread."""
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    history = []
    for m in state.values.get("messages", []):
        if m.type in ("human", "ai") and isinstance(m.content, str) and m.content.strip():
            history.append(
                {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            )
    # A pending confirmation question isn't in the messages yet, so add it.
    for task in state.tasks:
        for i in task.interrupts:
            history.append({"role": "assistant", "content": i.value})
    return history


if __name__ == "__main__":
    # Quick terminal test: python -m src.agent
    agent = build_agent()
    thread = str(uuid.uuid4())
    print("Stay Assistant (Ctrl+C to quit)")
    while True:
        q = input("\nYou: ")
        print("\nAssistant:", ask(agent, q, thread))
