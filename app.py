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
from src.tools.booking import list_bookings

st.set_page_config(page_title="Stay Assistant", page_icon="🏡")


@st.cache_resource
def get_agent():
    return build_agent()


def set_params(**kwargs):
    """Navigation is done through the URL: ?user=...&thread=..."""
    st.query_params.clear()
    for k, v in kwargs.items():
        st.query_params[k] = v


# ---------- Screen 1: choose a user ----------
def user_screen():
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


# ---------- Screen 2: sidebar ----------
def sidebar(user_id: str, thread_id: str):
    st.markdown(f"### 👤 {user_id}")
    if st.button("Switch user", use_container_width=True):
        set_params()
        st.rerun()

    # Collapsed + fixed height with scroll, so it never takes over the sidebar.
    prefs = get_user_preferences(user_id)
    with st.expander(f"Saved preferences ({len(prefs)})"):
        with st.container(height=150, border=False):
            for p in prefs:
                st.markdown(f"- {p}")
            if not prefs:
                st.caption("None yet.")

    bookings = list_bookings(user_id)
    with st.expander(f"My bookings ({len(bookings)})"):
        for b in bookings:
            st.markdown(f"- **{b['stay_name']}**  \n{b['check_in']} → {b['check_out']}")
        if not bookings:
            st.caption("None yet.")

    st.divider()
    if st.button("➕ New chat", use_container_width=True):
        set_params(user=user_id, thread=create_thread(user_id))
        st.rerun()

    # One plain text line per chat (like ChatGPT); the current one is highlighted.
    for t in list_threads(user_id):
        is_current = t["thread_id"] == thread_id
        if st.button(
            t["title"],
            key=t["thread_id"],
            use_container_width=True,
            type="secondary" if is_current else "tertiary",
        ):
            set_params(user=user_id, thread=t["thread_id"])
            st.rerun()


# ---------- Screen 2: chat ----------
def chat(agent, user_id: str, thread_id: str):
    st.title("Stay Assistant")

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


# ---------- Main ----------
user_id = st.query_params.get("user")
if not user_id:
    user_screen()
    st.stop()

agent = get_agent()

thread_id = st.query_params.get("thread")
if not thread_id or not get_thread(user_id, thread_id):
    user_threads = list_threads(user_id)
    thread_id = user_threads[0]["thread_id"] if user_threads else create_thread(user_id)
    set_params(user=user_id, thread=thread_id)

with st.sidebar:
    sidebar(user_id, thread_id)
chat(agent, user_id, thread_id)
