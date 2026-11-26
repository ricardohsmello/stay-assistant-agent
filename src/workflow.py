"""The agent's workflow, written as a graph. Read it top to bottom:

    START → plan → agent ─┬─ no tool call ──────────────→ END
                          ├─ search / availability → tools → agent
                          └─ book_stay → confirm (asks the user) → agent
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from src.config import CHAT_MODEL, EXTRACTION_MODEL
from src.tools import ALL_TOOLS
from src.tools.booking import book_stay

YES = {"yes", "y", "sim", "s", "ok", "confirm", "confirmo", "pode", "claro"}

PLAN_PROMPT = """Write a very short numbered plan (max 4 steps) to handle the user's request.
Available tools: search_stays, check_availability, book_stay (needs the user's confirmation).
Only plan what the user asked for. Output just the steps."""


class State(MessagesState):
    plan: str  # the plan for the current request


def build_workflow(prompt_fn, checkpointer):
    planner = ChatOpenAI(model=EXTRACTION_MODEL, temperature=0)
    # parallel_tool_calls=False: one tool call at a time, so booking is always handled alone
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0).bind_tools(ALL_TOOLS, parallel_tool_calls=False)

    # ---- Step 1: make a short plan from the user's last message ----
    def plan(state: State):
        last_user = next(m for m in reversed(state["messages"]) if m.type == "human")
        steps = planner.invoke([SystemMessage(PLAN_PROMPT), last_user]).content
        print(f"\n[plan]\n{steps}")
        return {"plan": steps}

    # ---- Step 2: the LLM thinks and either answers or asks for a tool ----
    def agent(state: State, config):
        return {"messages": [llm.invoke(prompt_fn(state, config))]}

    # ---- Step 3: decide where to go after the LLM spoke ----
    def route(state: State):
        calls = state["messages"][-1].tool_calls
        if not calls:
            return END
        return "confirm" if calls[0]["name"] == "book_stay" else "tools"

    # ---- Step 4a: safe tools (search, availability) just run ----
    tools = ToolNode([t for t in ALL_TOOLS if t.name != "book_stay"])

    # ---- Step 4b: booking pauses and asks the user first ----
    def confirm(state: State, config):
        call = state["messages"][-1].tool_calls[0]
        a = call["args"]
        question = (
            f"Do you confirm booking **{a['stay_name']}** "
            f"from {a['check_in']} to {a['check_out']}? (yes/no)"
        )
        answer = interrupt(question)  # graph stops here until the user replies

        if answer.strip().lower() in YES:
            result = book_stay.invoke(call, config)
        else:
            result = ToolMessage("The user declined the booking.", tool_call_id=call["id"])
        return {"messages": [result, AIMessage(question), HumanMessage(answer)]}

    # ---- Wiring ----
    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("confirm", confirm)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "agent")
    graph.add_conditional_edges("agent", route, ["tools", "confirm", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("confirm", "agent")
    return graph.compile(checkpointer=checkpointer)
