"""Streamlit chat UI for Stay Assistant. Run: streamlit run app.py"""
import streamlit as st

from src.agent import ask, build_agent, get_history
from src.conversations import (
    create_thread,
    create_user,
    get_thread,
    list_threads,
    list_users,
    normalize_user,
    touch_thread,
)
from src.memory import get_user_preferences

st.set_page_config(page_title="Stay Assistant", page_icon="🏡")


@st.cache_resource
def get_agent():
    return build_agent()


def set_params(**kwargs):
    st.query_params.clear()
    for k, v in kwargs.items():
        st.query_params[k] = v


# ---------- Screen 1: choose a user ----------
user_id = st.query_params.get("user")

if not user_id:
    st.title("🏡 Stay Assistant")
    st.subheader("Who is this?")

    existing = list_users()
    if existing:
        chosen = st.selectbox("Existing user", existing)
        if st.button("Continue", type="primary"):
            set_params(user=chosen)
            st.rerun()
        st.divider()

    with st.form("new_user"):
        new_name = st.text_input("Create new user")
        if st.form_submit_button("Create and continue") and new_name.strip():
            name = normalize_user(new_name)
            create_user(name)
            set_params(user=name)
            st.rerun()
    st.stop()

# ---------- Screen 2: the user's chat ----------
agent = get_agent()

thread_id = st.query_params.get("thread")
if not thread_id or not get_thread(user_id, thread_id):
    user_threads = list_threads(user_id)
    thread_id = user_threads[0]["thread_id"] if user_threads else create_thread(user_id)
    set_params(user=user_id, thread=thread_id)

with st.sidebar:
    st.markdown(f"### 👤 {user_id}")
    if st.button("Switch user", use_container_width=True):
        set_params()
        st.rerun()

    st.markdown("**Saved preferences**")
    prefs = get_user_preferences(user_id)
    if prefs:
        for p in prefs:
            st.markdown(f"- {p}")
    else:
        st.caption("None yet.")

    st.divider()
    if st.button("➕ New chat", use_container_width=True):
        set_params(user=user_id, thread=create_thread(user_id))
        st.rerun()

    for t in list_threads(user_id):
        is_current = t["thread_id"] == thread_id
        if st.button(
            t["title"],
            key=t["thread_id"],
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            set_params(user=user_id, thread=t["thread_id"])
            st.rerun()

st.title("🏡 Stay Assistant")

for msg in get_history(agent, thread_id):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What kind of stay are you looking for?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = ask(agent, prompt, thread_id, user_id)
        st.markdown(answer)

    touch_thread(thread_id, prompt)
