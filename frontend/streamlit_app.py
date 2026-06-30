"""
Streamlit frontend -- a chat interface over the FastAPI backend. Kept
deliberately simple (pure Python, no JS) since the point of this layer
is a clean demo, not a custom design system.
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Inbox Intelligence", page_icon="📬", layout="centered")
st.title("📬 Inbox Intelligence")
st.caption("Ask questions across your inbox. Answers are grounded in your actual emails (RAG).")

with st.sidebar:
    st.header("Setup")
    source = st.selectbox("Email source", ["sample", "gmail"], help="'sample' needs no setup. 'gmail' requires OAuth (see README).")

    if st.button("🔄 Sync emails", use_container_width=True):
        with st.spinner("Fetching and indexing emails..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/sync", json={"source": source}, timeout=300)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")
                resp = None

        if resp is not None:
            if resp.ok:
                data = resp.json()
                st.success(f"Indexed {data['documents_indexed']} chunks from '{data['source']}'")
            else:
                st.error(f"Sync failed: {resp.text}")

    st.divider()
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.caption(f"LLM provider: **{health['llm_provider']}**")
        st.caption(f"Configured source: **{health['data_source']}**")
    except requests.exceptions.RequestException:
        st.caption("⚠️ Backend not reachable")

if "history" not in st.session_state:
    st.session_state.history = []


CATEGORY_COLORS = {
    "Urgent": "🔴",
    "Finance": "💰",
    "Work": "💼",
    "Personal": "👤",
    "Promotional": "📢",
    "Travel": "✈️",
    "Social": "👥",
    "Other": "📧",
}


def _render_sources(sources, latency):
    with st.expander(f"📎 {len(sources)} source email(s) · {latency}s"):
        for i, s in enumerate(sources):
            category = s.get("category", "Other")
            emoji = CATEGORY_COLORS.get(category, "📧")
            st.markdown(f"**{s['subject']}**  \n*{s['sender']} — {s['date']}*")
            st.caption(f"{emoji} {category}")
            st.caption(s["snippet"])

            tone_key = f"tone_{i}_{s['subject']}"
            draft_key = f"draft_{i}_{s['subject']}"
            btn_key = f"btn_{i}_{s['subject']}"

            tone = st.selectbox(
                "Reply tone",
                ["professional", "friendly", "brief"],
                key=tone_key,
                label_visibility="collapsed",
            )
            if st.button("✏️ Draft a reply", key=btn_key, use_container_width=False):
                with st.spinner("Drafting reply..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/draft-reply",
                            json={
                                "subject": s["subject"],
                                "sender": s["sender"],
                                "date": s["date"],
                                "snippet": s["snippet"],
                                "tone": tone,
                            },
                            timeout=60,
                        )
                        if resp.ok:
                            st.session_state[draft_key] = resp.json()["draft"]
                        else:
                            st.error(f"Draft failed: {resp.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not reach backend: {e}")

            if draft_key in st.session_state:
                st.text_area(
                    "Draft reply",
                    value=st.session_state[draft_key],
                    height=180,
                    key=f"ta_{draft_key}",
                    label_visibility="collapsed",
                )

            st.markdown("---")


for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"].replace("$", "\\$"))
        _render_sources(entry["sources"], entry["latency"])

question = st.chat_input("Ask something about your inbox...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/ask", json={"question": question}, timeout=300)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")
                resp = None

        if resp is not None:
            if resp.ok:
                data = resp.json()
                st.write(data["answer"].replace("$", "\\$"))
                _render_sources(data["sources"], data["latency_seconds"])
                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": data["answer"],
                        "sources": data["sources"],
                        "latency": data["latency_seconds"],
                    }
                )
            else:
                st.error(f"Query failed: {resp.text}")