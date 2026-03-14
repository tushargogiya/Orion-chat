import os
import queue
import uuid

import streamlit as st

# Inject Streamlit Cloud secrets into os.environ BEFORE importing backend
# (backend reads keys via os.getenv() at import time)
for _k, _v in st.secrets.items():
    if _k not in os.environ:
        os.environ[_k] = str(_v)

from langgraph_backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ========================= Page Config =========================
st.set_page_config(
    page_title="AI Agent Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================= Custom CSS =========================
st.markdown("""
<style>
    /* Hide default streamlit header padding */
    .main .block-container { padding-top: 1.5rem; max-width: 860px; }

    /* Feature badge pills */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin: 2px 2px;
    }
    .b-green  { background:#14532d; color:#4ade80; border:1px solid #166534; }
    .b-blue   { background:#1e3a5f; color:#60a5fa; border:1px solid #1d4ed8; }
    .b-purple { background:#2e1065; color:#c084fc; border:1px solid #7c3aed; }
    .b-orange { background:#431407; color:#fb923c; border:1px solid #c2410c; }
    .b-pink   { background:#4a044e; color:#f0abfc; border:1px solid #a21caf; }
    .b-cyan   { background:#083344; color:#67e8f9; border:1px solid #0e7490; }

    /* Thread list item */
    .thread-btn {
        display:flex; align-items:center; gap:8px;
        padding: 7px 10px; border-radius:8px;
        border:1px solid rgba(255,255,255,0.08);
        background:rgba(255,255,255,0.04);
        color:rgba(255,255,255,0.75);
        font-size:0.78rem; cursor:pointer; margin-bottom:4px;
        transition: background 0.15s;
    }
    .thread-btn:hover { background:rgba(255,255,255,0.12); }
    .thread-active {
        background:rgba(96,165,250,0.15) !important;
        border-color:rgba(96,165,250,0.5) !important;
        color:#93c5fd !important;
    }

    /* Current thread pill in header */
    .thread-pill {
        display:inline-block;
        padding:3px 12px; border-radius:999px;
        background:rgba(96,165,250,0.15);
        border:1px solid rgba(96,165,250,0.4);
        color:#93c5fd; font-size:0.75rem; font-weight:600;
        vertical-align:middle; margin-left:10px;
    }

    /* Footer text */
    .footer-text {
        font-size:0.65rem; color:rgba(255,255,255,0.3);
        text-align:center; line-height:1.6;
    }
</style>
""", unsafe_allow_html=True)


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def short_id(thread_id: str) -> str:
    return str(thread_id)[:8].upper()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.rerun()


def add_thread(thread_id: str):
    tid = str(thread_id)
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)


def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [str(t) for t in retrieve_all_threads()]

add_thread(st.session_state["thread_id"])


# ============================ Sidebar ============================
with st.sidebar:
    st.markdown("## 🤖 AI Agent Chatbot")
    st.caption("LangGraph · Azure GPT · Streamlit")

    st.divider()

    # Feature showcase
    st.markdown("**Capabilities**")
    st.markdown("""
    <div style="margin-bottom:14px; line-height:2;">
        <span class="badge b-green">✦ Persistence</span>
        <span class="badge b-blue">✦ Resume Chat</span>
        <span class="badge b-purple">✦ MCP Server</span>
        <span class="badge b-orange">✦ Web Search</span>
        <span class="badge b-pink">✦ HITL</span>
        <span class="badge b-cyan">✦ Streaming</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("➕  New Conversation", use_container_width=True, type="primary"):
        reset_chat()

    st.markdown("**Chat History**")
    active_tid = str(st.session_state["thread_id"])

    for thread_id in st.session_state["chat_threads"][::-1]:
        tid = str(thread_id)
        is_active = tid == active_tid
        label = f"{'▶  ' if is_active else '💬  '}Chat #{short_id(tid)}"
        btn_style = "primary" if is_active else "secondary"

        if st.button(label, key=f"t_{tid}", use_container_width=True, type=btn_style):
            st.session_state["thread_id"] = tid
            messages = load_conversation(tid)
            temp = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    temp.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    temp.append({"role": "assistant", "content": msg.content})
            st.session_state["message_history"] = temp
            st.rerun()

    st.divider()
    st.markdown("""
    <div class="footer-text">
        Built with LangGraph + LangChain<br>
        SQLite Persistence · Async Streaming<br>
        MCP Protocol · Azure OpenAI
    </div>
    """, unsafe_allow_html=True)


# ============================ Main UI ============================
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(
        f'<h2 style="margin:0 0 4px 0; display:inline;">💬 Chat'
        f'<span class="thread-pill">Thread: {short_id(active_tid)}</span></h2>',
        unsafe_allow_html=True,
    )
with col2:
    if st.button("🗑️ Clear", help="Start a new conversation"):
        reset_chat()

st.markdown("---")

# Welcome message when conversation is empty
if not st.session_state["message_history"]:
    with st.chat_message("assistant"):
        st.markdown("""
**👋 Hello! I'm an AI agent with real capabilities — here's what I can do:**

| Feature | Description |
|---|---|
| 🔍 **Web Search** | Real-time web search via DuckDuckGo |
| 📈 **Stock Prices** | Live stock data (e.g. *"What's AAPL price?"*) |
| 🧾 **Expense Tracking** | Connected to an MCP expense server |
| 💾 **Persistent Memory** | Conversations saved & resumable across sessions |
| 🔄 **Streaming** | Token-by-token response streaming |
| 👤 **Human-in-the-Loop** | Approval flow for sensitive actions |

*Try asking: "Search for latest AI news" or "Get me TSLA stock price"*
        """)

# Render existing messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ========================= Chat Input =========================
user_input = st.chat_input("Ask me anything…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": str(st.session_state["thread_id"])},
        "metadata": {"thread_id": str(st.session_state["thread_id"])},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Running `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Running `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool completed", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
