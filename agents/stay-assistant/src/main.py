"""Entry point for Atlas Agent Engine (see `entrypoint` in agent.yaml)."""
from agent_engine_sdk_langgraph import App
from langchain_openai import ChatOpenAI

from src.config import CHAT_MODEL
from src.tools import ALL_TOOLS
from src.workflow import build_workflow

app = App(app_name="stay-assistant")

# Register the tools on the platform.
for tool_fn in ALL_TOOLS:
    app.tool()(tool_fn)


def get_memories(query: str) -> str:
    """Long-term memory: what the platform learned about this user in past conversations."""
    try:
        context = app.memory.build_context(query=query, max_tokens=1000).formatted_context
    except Exception as e:  # memory must never break the chat
        print(f"[memory] unavailable: {e}")
        return "none yet"
    if isinstance(context, list):
        context = "\n".join(str(c) for c in context)
    return context.strip() or "none yet"


@app.entrypoint
def build_agent():
    llm = app.llm(ChatOpenAI(model=CHAT_MODEL, temperature=0))
    llm_with_tools = llm.bind_tools(app.get_tool_schemas())  # lets the LLM call our tools
    return build_workflow(llm, llm_with_tools, app.get_tools(), app.checkpointer(), get_memories)


app.run()
