"""
Study & Task Planner Agent  -  core code (used by the notebook, main.py and app.py)

How it works in one sentence:
the AI model (Gemini) reads your message, decides if it needs a tool, the tool runs,
and the AI uses the result to answer. Dangerous tools (delete) pause and ask YOU first.
"""

# %% [markdown-cell: Imports]
import os
from typing import Annotated, TypedDict

import requests
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

# %% [markdown-cell: Tools - task list (custom Python tools, no external API)]
# The task list lives in normal Python memory (a list of dictionaries).
TASKS = []
_next_id = 1


@tool
def add_task(title: str) -> str:
    """Add a new task to the to-do list. Use this when the user wants to remember or plan something."""
    global _next_id
    TASKS.append({"id": _next_id, "title": title, "done": False})
    _next_id += 1
    return f"Task added: #{_next_id - 1} '{title}'"


@tool
def list_tasks() -> str:
    """Show all tasks in the to-do list with their id and whether they are done."""
    if not TASKS:
        return "The task list is empty."
    lines = [f"#{t['id']} [{'x' if t['done'] else ' '}] {t['title']}" for t in TASKS]
    return "\n".join(lines)


@tool
def complete_task(task_id: int) -> str:
    """Mark one task as done, using its id number (see list_tasks)."""
    for t in TASKS:
        if t["id"] == task_id:
            t["done"] = True
            return f"Task #{task_id} marked as done."
    return f"No task with id {task_id}."


@tool
def delete_task(task_id: int) -> str:
    """Permanently delete one task by its id number. DESTRUCTIVE: needs human approval."""
    for t in TASKS:
        if t["id"] == task_id:
            TASKS.remove(t)
            return f"Task #{task_id} deleted."
    return f"No task with id {task_id}."


@tool
def clear_all_tasks() -> str:
    """Permanently delete ALL tasks from the list. DESTRUCTIVE: needs human approval."""
    count = len(TASKS)
    TASKS.clear()
    return f"All {count} tasks deleted."


# %% [markdown-cell: Tools - external APIs (free, no API key needed)]
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (uses the free Open-Meteo API)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Could not find a city called '{city}'."
        place = geo["results"][0]
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,wind_speed_10m,precipitation",
            },
            timeout=10,
        ).json()["current"]
        return (
            f"Weather in {place['name']}, {place.get('country', '')}: "
            f"{weather['temperature_2m']} C, wind {weather['wind_speed_10m']} km/h, "
            f"precipitation {weather['precipitation']} mm."
        )
    except Exception as error:  # network problems, bad response, etc.
        return f"Weather service error: {error}"


@tool
def search_wikipedia(topic: str) -> str:
    """Search Wikipedia and return a short summary of a topic (free Wikipedia API)."""
    try:
        headers = {"User-Agent": "StudyPlannerAgent/1.0 (student project)"}
        found = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
            headers=headers,
            timeout=10,
        ).json()
        if not found[1]:
            return f"No Wikipedia page found for '{topic}'."
        title = found[1][0]
        summary = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_"),
            headers=headers,
            timeout=10,
        ).json()
        return f"{summary['title']}: {summary['extract']}"
    except Exception as error:
        return f"Wikipedia error: {error}"


SAFE_TOOLS = [add_task, list_tasks, complete_task, get_weather, search_wikipedia]
DESTRUCTIVE_TOOLS = [delete_task, clear_all_tasks]
ALL_TOOLS = SAFE_TOOLS + DESTRUCTIVE_TOOLS
DESTRUCTIVE_NAMES = {t.name for t in DESTRUCTIVE_TOOLS}
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

# %% [markdown-cell: State and Graph]
SYSTEM_PROMPT = (
    "You are a friendly Study & Task Planner assistant. You manage a to-do list, "
    "check the weather and look things up on Wikipedia. "
    "When the user asks to delete tasks, call the delete tool directly - the system "
    "will ask the user for approval, so do NOT ask for confirmation yourself. "
    "If a tool says an action was cancelled, tell the user nothing was changed."
)


class State(TypedDict):
    # 'messages' is the whole conversation. add_messages appends new messages to it.
    messages: Annotated[list, add_messages]


def build_graph(llm, checkpointer=None):
    """Build the agent graph. 'llm' is the chat model, 'checkpointer' stores memory."""
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def assistant(state: State):
        """The AI reads the conversation and either answers or asks for a tool call."""
        reply = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    def route(state: State):
        """Decide the next step: finish, run safe tools, or ask for approval first."""
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None)
        if not calls:
            return END
        if any(c["name"] in DESTRUCTIVE_NAMES for c in calls):
            return "approval"
        return "safe_tools"

    def approval(state: State):
        """Pause and ask the human before any destructive tool runs."""
        calls = state["messages"][-1].tool_calls

        # Step 1: ask the human about every destructive call.
        # (interrupt() pauses the graph; the node restarts when the human answers,
        #  so nothing with side effects may happen before this loop.)
        approved = {}
        for call in calls:
            if call["name"] in DESTRUCTIVE_NAMES:
                answer = interrupt(
                    {
                        "action": call["name"],
                        "args": call["args"],
                        "question": describe_action(call),
                    }
                )
                approved[call["id"]] = str(answer).strip().lower() in ("yes", "y", "approve", "ok")

        # Step 2: run the allowed calls, skip the denied ones.
        results = []
        for call in calls:
            if call["name"] in DESTRUCTIVE_NAMES and not approved[call["id"]]:
                text = "Action cancelled: the user denied approval. Nothing was changed."
            else:
                text = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            results.append(ToolMessage(content=str(text), tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("safe_tools", ToolNode(SAFE_TOOLS))
    builder.add_node("approval", approval)
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", route, ["safe_tools", "approval", END])
    builder.add_edge("safe_tools", "assistant")
    builder.add_edge("approval", "assistant")
    return builder.compile(checkpointer=checkpointer)


def describe_action(call) -> str:
    """Human-readable text for the approval question."""
    if call["name"] == "delete_task":
        task_id = call["args"].get("task_id")
        title = next((t["title"] for t in TASKS if t["id"] == task_id), "unknown task")
        return f"Delete task #{task_id} ('{title}')?"
    if call["name"] == "clear_all_tasks":
        return f"Delete ALL {len(TASKS)} tasks?"
    return f"Run {call['name']}?"


# %% [markdown-cell: Create the model and the agent (memory = MemorySaver)]
def make_llm():
    """Gemini chat model. Reads GOOGLE_API_KEY from the environment."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    # If this model name is rejected, open Google AI Studio and copy a current free model name.
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)


def create_agent(llm=None):
    """Create the agent with in-memory conversation memory."""
    return build_graph(llm or make_llm(), MemorySaver())


# %% [markdown-cell: Helper functions for chat apps (CLI and Streamlit)]
def message_text(message) -> str:
    """Get plain text out of a message (Gemini sometimes returns a list of parts)."""
    content = message.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def pending_approval(agent, config):
    """If the agent is paused waiting for approval, return the question info, else None."""
    state = agent.get_state(config)
    if state.next and state.tasks and state.tasks[0].interrupts:
        return state.tasks[0].interrupts[0].value
    return None


def chat_turn(agent, user_text, thread_id):
    """Send one user message. Returns ('reply', text) or ('approval', info)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [("user", user_text)]}, config)
    return _outcome(agent, result, config)


def resume_turn(agent, decision, thread_id):
    """Answer a pending approval question with 'yes' or 'no'."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(Command(resume=decision), config)
    return _outcome(agent, result, config)


def _outcome(agent, result, config):
    info = pending_approval(agent, config)
    if info:
        return "approval", info
    last_ai = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
    return "reply", message_text(last_ai) if last_ai else ""
