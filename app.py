"""Streamlit chat UI for the Study & Task Planner Agent.  Run:  streamlit run app.py"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Study & Task Planner Agent", page_icon="📝")
st.title("📝 Study & Task Planner Agent")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is missing. Copy .env.example to .env and paste your key.")
    st.stop()

from agent import TASKS, chat_turn, create_agent, resume_turn  # noqa: E402

THREAD_ID = "streamlit-session"

# st.session_state survives reruns, so the agent (and its memory) is created only once.
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()
    st.session_state.history = []  # list of (role, text)
    st.session_state.pending = None  # approval info while the agent is paused

agent = st.session_state.agent


def handle(kind, content):
    if kind == "approval":
        st.session_state.pending = content
    else:
        st.session_state.pending = None
        st.session_state.history.append(("assistant", content))


# Sidebar: live view of the task list
with st.sidebar:
    st.header("Current tasks")
    if TASKS:
        for t in TASKS:
            st.write(f"{'✅' if t['done'] else '⬜'} #{t['id']} {t['title']}")
    else:
        st.caption("No tasks yet.")

# Show the conversation so far
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

# Approval step: the agent is paused until the user clicks a button
if st.session_state.pending:
    st.warning(f"Approval needed: {st.session_state.pending['question']}")
    yes_col, no_col = st.columns(2)
    if yes_col.button("✅ Yes, do it"):
        handle(*resume_turn(agent, "yes", THREAD_ID))
        st.rerun()
    if no_col.button("❌ No, cancel"):
        handle(*resume_turn(agent, "no", THREAD_ID))
        st.rerun()
elif prompt := st.chat_input("Ask me to add a task, check weather, search Wikipedia..."):
    st.session_state.history.append(("user", prompt))
    with st.spinner("Thinking..."):
        handle(*chat_turn(agent, prompt, THREAD_ID))
    st.rerun()
