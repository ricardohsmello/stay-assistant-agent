"""Streamlit chat UI for Stay Assistant. Run: streamlit run app.py"""
import uuid

import streamlit as st

from src.agent import ask, build_agent

st.set_page_config(page_title="Stay Assistant", page_icon="🏡")
st.title("🏡 Stay Assistant")


@st.cache_resource
def get_agent():
    return build_agent()


agent = get_agent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What kind of stay are you looking for??"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = ask(agent, prompt, st.session_state.thread_id)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
