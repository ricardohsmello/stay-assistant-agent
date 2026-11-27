"""The agent's workflow, written as a graph. Read it top to bottom:

    START → plan → agent ─┬─ no tool call ────────────────────→ END
                          ├─ search / availability ───────────→ tools → agent
                          └─ book_stay → confirm (asks user) ─→ tools → agent
                                                  └─ declined ──────→ agent
"""
from datetime import date

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

YES = {"yes", "y", "sim", "s", "ok", "confirm", "confirmo", "pode", "claro"}

SYSTEM_PROMPT = """You are Stay Assistant, an assistant that helps users find Airbnb \
stays that match what they are looking for. Today is {today}.
- Use the `search_stays` tool to search the database; never invent listings.
- Always search right away with whatever the user has said so far; never ask questions before searching.
- Keep answers short, highlighting name, location, price per night, capacity, and why it matches.

Booking:
- To book, you need the stay and the check-in/check-out dates; ask for the dates if missing. \
Use the current year (or the next one if the date already passed).
- Call `check_availability` first, then call `book_stay`.
- Do NOT ask the user for confirmation yourself: the system asks automatically when you call `book_stay`.

What we already know about this user (from past conversations):
{memories}

Plan for the current request:
{plan}"""

PLAN_PROMPT = """Write a very short numbered plan (max 4 steps) to handle the user's request.
Available tools: search_stays, check_availability, book_stay (the system asks the user to confirm).
Only plan what the user asked for. Never plan to ask questions before searching: if the user
wants a stay, step 1 is always search_stays with whatever was said. Output just the steps."""


class State(MessagesState):
    plan: str  # the plan for the current request


def build_workflow(llm, llm_with_tools, tools, checkpointer, get_memories):
    # ---- Step 1: make a short plan from the user's last message ----
    def plan(state: State):
        last_user = next(m for m in reversed(state["messages"]) if m.type == "human")
        steps = llm.invoke([SystemMessage(PLAN_PROMPT), last_user]).content
        return {"plan": steps}

    # ---- Step 2: the LLM thinks and either answers or asks for a tool ----
    def agent(state: State):
        last_user = next(m for m in reversed(state["messages"]) if m.type == "human")
        system = SYSTEM_PROMPT.format(
            today=date.today().isoformat(),
            memories=get_memories(last_user.content),
            plan=state.get("plan", ""),
        )
        return {"messages": [llm_with_tools.invoke([SystemMessage(system)] + state["messages"])]}

    # ---- Step 3: decide where to go after the LLM spoke ----
    def route(state: State):
        calls = state["messages"][-1].tool_calls
        if not calls:
            return END
        return "confirm" if any(c["name"] == "book_stay" for c in calls) else "tools"

    # ---- Step 4a: tools (search, availability, booking) run here ----
    tool_node = ToolNode(tools)

    # ---- Step 4b: booking pauses and asks the user first ----
    def confirm(state: State):
        call = next(c for c in state["messages"][-1].tool_calls if c["name"] == "book_stay")
        a = call["args"]
        answer = interrupt(  # graph stops here until the user replies
            f"Do you confirm booking {a.get('stay_name')} "
            f"from {a.get('check_in')} to {a.get('check_out')}? (yes/no)"
        )
        if answer.strip().lower() in YES:
            return Command(goto="tools")
        declined = [
            ToolMessage("The user declined the booking.", tool_call_id=c["id"])
            for c in state["messages"][-1].tool_calls
        ]
        return Command(goto="agent", update={"messages": declined})

    # ---- Wiring ----
    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("confirm", confirm)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "agent")
    graph.add_conditional_edges("agent", route, ["tools", "confirm", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
