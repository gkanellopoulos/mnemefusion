"""
MnemeFusion Chat Demo — Manual Testing UI

A Streamlit chat app for manually testing MnemeFusion's memory capabilities.
Each user gets their own .mfdb file (long-term brain) and can have multiple
conversations. Starting a new conversation clears the chat thread but keeps
all memories — so you can test cross-conversation recall.

LLM entity extraction is enabled automatically when the GGUF model is found,
which is REQUIRED for the entity dimension to work. Without it, the library
only uses semantic + BM25 (2 of 5 dimensions) and retrieval quality is poor.

Run with:
    streamlit run apps/chat_demo.py

Requirements:
    pip install streamlit sentence-transformers openai python-dotenv mnemefusion
"""

import os
import time
import json
import gc
import uuid
import datetime

# Force CPU-only LLM inference — prevents 4GB GPU from freezing the desktop.
# The model runs on CPU (~30-60s per extraction) but entity profiles actually work.
os.environ["MNEMEFUSION_GPU_LAYERS"] = "0"

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from mnemefusion import Memory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "chat_data"
DATA_DIR.mkdir(exist_ok=True)

# Workspace root (for resolving model paths and DLLs)
WORKSPACE_ROOT = Path(__file__).parent.parent

EMBEDDING_DIM = 768
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

# LLM entity extraction model — optional, enables entity dimension
# Auto-detected: tries Phi-4-mini (recommended), falls back to Qwen3-4B
_MODEL_CANDIDATES = [
    WORKSPACE_ROOT / "models" / "phi-4-mini" / "Phi-4-mini-instruct-Q4_K_M.gguf",
    WORKSPACE_ROOT / "models" / "qwen3-4b" / "Qwen3-4B-Instruct-2507.Q4_K_M.gguf",
]
GGUF_MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[0])

SYSTEM_PROMPT = """\
You are a helpful assistant with access to a personal memory system.
Use the provided memory context to give personalized, accurate responses.
If the context contains relevant information, use it naturally in your reply.
If the context doesn't help, just respond normally — don't mention that memories were empty.\
"""

# ---------------------------------------------------------------------------
# Cached resources (shared across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_encoder():
    # Force CPU for embeddings — GPU is reserved for LLM entity extraction
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@st.cache_resource
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def embed(text: str) -> list[float]:
    encoder = load_encoder()
    return encoder.encode(text).tolist()


# ---------------------------------------------------------------------------
# Memory management — persistent per-user Memory object in session_state
# ---------------------------------------------------------------------------

def _state_key(username: str) -> str:
    return f"_mem_{username}"


def get_memory(username: str) -> Memory:
    """Get or create a persistent Memory object for the user.

    The Memory object (with LLM model loaded) lives in st.session_state
    so it survives Streamlit reruns without reloading the 2.3GB model.
    """
    key = _state_key(username)

    if key in st.session_state and st.session_state[key] is not None:
        return st.session_state[key]

    # Close any previously open Memory for a different user
    _close_all_memories_except(username)

    path = str(DATA_DIR / f"{username}.mfdb")
    mem = Memory(path, {"embedding_dim": EMBEDDING_DIM})
    mem.set_embedding_fn(embed)
    mem.set_user_entity(username)

    # LLM entity extraction — runs on CPU to avoid freezing 4GB GPU
    # Slower (~30-60s per message) but entity profiles actually work.
    if GGUF_MODEL_PATH.exists():
        try:
            mem.enable_llm_entity_extraction(str(GGUF_MODEL_PATH), extraction_passes=1)
            st.session_state["_llm_extraction_active"] = True
        except Exception as e:
            st.session_state["_llm_extraction_active"] = False
            st.session_state["_llm_extraction_error"] = str(e)
    else:
        st.session_state["_llm_extraction_active"] = False

    st.session_state[key] = mem
    return mem


def _close_all_memories_except(keep_username: str):
    """Close any Memory objects for other users."""
    keep_key = _state_key(keep_username)
    to_remove = []
    for k, v in st.session_state.items():
        if k.startswith("_mem_") and k != keep_key and v is not None:
            try:
                v.close()
            except Exception:
                pass
            to_remove.append(k)
    for k in to_remove:
        st.session_state[k] = None
    gc.collect()


def close_user_memory(username: str):
    """Explicitly close a user's Memory (e.g., before deleting)."""
    key = _state_key(username)
    if key in st.session_state and st.session_state[key] is not None:
        try:
            st.session_state[key].close()
        except Exception:
            pass
        st.session_state[key] = None
    gc.collect()


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

def list_users() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.mfdb"))


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------

def conversations_dir(username: str) -> Path:
    d = DATA_DIR / f"{username}_convos"
    d.mkdir(exist_ok=True)
    return d


def list_conversations(username: str) -> list[dict]:
    d = conversations_dir(username)
    convos = []
    for f in d.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        convos.append({
            "id": f.stem,
            "title": data.get("title", "Untitled"),
            "created_at": data.get("created_at", ""),
            "simulated_date": data.get("simulated_date", ""),
        })
    convos.sort(key=lambda c: c["created_at"], reverse=True)
    return convos


def create_conversation(username: str) -> str:
    convo_id = uuid.uuid4().hex[:8]
    d = conversations_dir(username)
    data = {
        "title": "New conversation",
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
        "messages": [],
    }
    (d / f"{convo_id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return convo_id


def load_conversation(username: str, convo_id: str) -> dict:
    f = conversations_dir(username) / f"{convo_id}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {"title": "New conversation", "created_at": "", "messages": []}


def save_conversation(username: str, convo_id: str, data: dict):
    f = conversations_dir(username) / f"{convo_id}.json"
    f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_conversation(username: str, convo_id: str):
    f = conversations_dir(username) / f"{convo_id}.json"
    if f.exists():
        f.unlink()


# ---------------------------------------------------------------------------
# Core chat logic
# ---------------------------------------------------------------------------

def process_chat_turn(username: str, convo_id: str, prompt: str,
                      chat_messages: list[dict], client,
                      simulated_ts: float | None = None):
    """Process one chat turn: retrieve FIRST, then store user msg, generate response."""
    mem = get_memory(username)

    # Retrieve context BEFORE storing the new message (avoids self-match)
    t0 = time.time()
    intent, context_str, raw_results = _retrieve_context(mem, prompt)
    latency_ms = (time.time() - t0) * 1000

    # NOW store user message in long-term memory (only user messages, not assistant)
    mem.reserve_capacity(mem.count() + 10)
    mem.add(
        content=prompt,
        metadata={"role": "user", "conversation": convo_id, "speaker": username},
        timestamp=simulated_ts,
    )

    # Generate response
    if client:
        response_text = _generate_response(client, prompt, context_str, chat_messages)
    else:
        if context_str.strip():
            response_text = (
                "[No API key — showing retrieval only]\n\n"
                f"Retrieved context:\n{context_str}"
            )
        else:
            response_text = "[No API key — no memories retrieved yet]"

    # Assistant responses are NOT stored in memory — they're paraphrases
    # of user content and add noise that buries the actual facts

    retrieval_info = {
        "intent": intent,
        "count": len(raw_results),
        "latency_ms": latency_ms,
        "results": raw_results,
    }
    return response_text, retrieval_info


def _retrieve_context(mem, query: str):
    """Query the memory and return (intent, context_str, raw_results)."""
    if mem.count() == 0:
        return "none", "", []

    intent, results, profile_ctx = mem.query(query, limit=10)

    context_lines = list(profile_ctx)  # Start with profile context
    raw_results = []
    for r in results:
        mem_dict, scores = r
        content = mem_dict.get("content", "")
        context_lines.append(content)
        raw_results.append({
            "content": content[:200],
            "fused_score": round(scores.get("fused_score", 0.0), 4),
            "semantic_score": round(scores.get("semantic_score", 0.0), 4),
            "entity_score": round(scores.get("entity_score", 0.0), 4),
            "bm25_score": round(scores.get("bm25_score", 0.0), 4),
            "temporal_score": round(scores.get("temporal_score", 0.0), 4),
        })

    context_str = "\n".join(f"- {line}" for line in context_lines)
    return intent, context_str, raw_results


def _generate_response(client, query: str, context: str,
                       chat_messages: list[dict]) -> str:
    """Generate an LLM response using retrieved memory context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in chat_messages[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if context.strip():
        user_msg = f"Memory context:\n{context}\n\nUser: {query}"
    else:
        user_msg = query

    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="MnemeFusion Chat Demo", layout="wide")

    # --- Initialize session state ---
    if "selected_user" not in st.session_state:
        st.session_state.selected_user = None
    if "selected_convo" not in st.session_state:
        st.session_state.selected_convo = None

    # --- Sidebar ---
    with st.sidebar:
        st.header("MnemeFusion Chat")

        # -- Status indicators --
        llm_active = st.session_state.get("_llm_extraction_active", None)
        if llm_active is True:
            st.success("Entity extraction: ON (CPU)", icon="🧠")
        elif llm_active is False:
            err = st.session_state.get("_llm_extraction_error", "")
            if err:
                st.error(f"Entity extraction: FAILED\n{err}", icon="⚠️")
            else:
                st.warning("Entity extraction: OFF — model not found", icon="⚠️")
                st.caption(f"Expected: {GGUF_MODEL_PATH}")

        st.divider()

        # -- User section --
        st.subheader("User")
        new_user = st.text_input("New user", placeholder="username", label_visibility="collapsed")
        col_create, _ = st.columns([1, 1])
        with col_create:
            if st.button("Create user", use_container_width=True) and new_user.strip():
                username = new_user.strip().lower().replace(" ", "_")
                if username in list_users():
                    st.warning(f"'{username}' already exists.")
                else:
                    # Initialize the Memory (loads model on first access)
                    mem = get_memory(username)
                    st.session_state.selected_user = username
                    st.session_state.selected_convo = None
                    st.rerun()

        users = list_users()
        if not users:
            st.info("Create a user to get started.")
            return

        # User selector
        current_user = st.session_state.selected_user
        if current_user not in users:
            current_user = users[0]
            st.session_state.selected_user = current_user

        idx = users.index(current_user) if current_user in users else 0
        selected = st.selectbox("Switch user", users, index=idx, label_visibility="collapsed")
        if selected != st.session_state.selected_user:
            st.session_state.selected_user = selected
            st.session_state.selected_convo = None
            st.rerun()

        mem = get_memory(selected)
        st.caption(f"Memories in brain: **{mem.count()}**")

        st.divider()

        # -- Conversations section --
        st.subheader("Conversations")

        if st.button("New conversation", use_container_width=True, type="primary"):
            convo_id = create_conversation(selected)
            st.session_state.selected_convo = convo_id
            st.rerun()

        convos = list_conversations(selected)
        if not convos:
            st.caption("No conversations yet.")
        else:
            for c in convos:
                sim = c.get("simulated_date", "")
                date_tag = f" [{sim}]" if sim else ""
                label = f"{c['title']}{date_tag}"
                is_active = c["id"] == st.session_state.selected_convo
                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    if st.button(
                        label,
                        key=f"convo_{c['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state.selected_convo = c["id"]
                        st.rerun()
                with col_del:
                    if st.button("x", key=f"del_{c['id']}"):
                        delete_conversation(selected, c["id"])
                        if st.session_state.selected_convo == c["id"]:
                            st.session_state.selected_convo = None
                        st.rerun()

        st.divider()

        # -- Danger zone --
        if st.button("Delete user & all data", type="secondary"):
            close_user_memory(selected)
            db_path = DATA_DIR / f"{selected}.mfdb"
            convos_dir_path = conversations_dir(selected)
            if db_path.exists():
                db_path.unlink()
            import shutil
            if convos_dir_path.exists():
                shutil.rmtree(convos_dir_path, ignore_errors=True)
            st.session_state.selected_user = None
            st.session_state.selected_convo = None
            st.rerun()

    # --- Main chat area ---
    username = st.session_state.selected_user
    convo_id = st.session_state.selected_convo

    if not username:
        return

    if not convo_id:
        st.markdown(
            "### Select a conversation or start a new one\n\n"
            "Each conversation has its own chat thread, but all memories "
            "are stored in one shared brain per user. Start a new conversation "
            "to test cross-conversation recall."
        )
        return

    client = get_openai_client()
    if client is None:
        st.error("Set OPENAI_API_KEY in apps/.env to enable LLM responses.")

    # Load conversation
    convo_data = load_conversation(username, convo_id)
    chat_messages = convo_data.get("messages", [])

    # --- Simulated date picker ---
    # Lock once conversation has at least one assistant response
    has_responses = any(m["role"] == "assistant" for m in chat_messages)
    saved_date_str = convo_data.get("simulated_date")

    if has_responses and saved_date_str:
        # Locked — show the date but don't allow changes
        sim_date = datetime.date.fromisoformat(saved_date_str)
        st.info(f"Simulated date: **{sim_date.strftime('%B %d, %Y')}** (locked)")
    else:
        # Editable — let user pick a date
        default_date = (datetime.date.fromisoformat(saved_date_str)
                        if saved_date_str else datetime.date.today())
        sim_date = st.date_input(
            "Simulated date for this conversation",
            value=default_date,
            help="Memories stored in this conversation will use this date as their timestamp. "
                 "Set to a past date to simulate long-term memory. Locks after first response.",
        )
        # Persist the chosen date immediately
        if str(sim_date) != saved_date_str:
            convo_data["simulated_date"] = str(sim_date)
            save_conversation(username, convo_id, convo_data)

    # Convert to Unix timestamp (noon on that day)
    sim_dt = datetime.datetime.combine(sim_date, datetime.time(12, 0))
    simulated_ts = sim_dt.timestamp()

    # Display messages
    for i, msg in enumerate(chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "retrieval" in msg:
                ret = msg["retrieval"]
                count = ret.get("count", 0)
                latency = ret.get("latency_ms", 0)
                intent = ret.get("intent", "N/A")

                with st.popover(f"ℹ {count} memories · {latency:.0f}ms"):
                    st.markdown(f"**Intent:** {intent}")
                    st.markdown(f"**Retrieved:** {count} memories")
                    st.markdown(f"**Latency:** {latency:.0f}ms")

                    if ret.get("results"):
                        st.divider()
                        for j, r in enumerate(ret["results"], 1):
                            st.code(
                                f"[{j}] fused={r['fused_score']:.4f}  "
                                f"sem={r['semantic_score']:.4f}  "
                                f"ent={r['entity_score']:.4f}  "
                                f"bm25={r['bm25_score']:.4f}  "
                                f"temp={r['temporal_score']:.4f}",
                                language=None,
                            )
                            st.caption(r["content"])
                    else:
                        st.caption("No memories retrieved.")

    # Chat input
    if prompt := st.chat_input(f"Message as {username}..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking... (CPU entity extraction ~30-60s on first messages)"):
            response_text, retrieval_info = process_chat_turn(
                username, convo_id, prompt, chat_messages, client,
                simulated_ts=simulated_ts,
            )

        with st.chat_message("assistant"):
            st.markdown(response_text)

            ret = retrieval_info
            count = ret.get("count", 0)
            latency = ret.get("latency_ms", 0)
            intent = ret.get("intent", "N/A")

            with st.popover(f"ℹ {count} memories · {latency:.0f}ms"):
                st.markdown(f"**Intent:** {intent}")
                st.markdown(f"**Retrieved:** {count} memories")
                st.markdown(f"**Latency:** {latency:.0f}ms")

                if ret.get("results"):
                    st.divider()
                    for j, r in enumerate(ret["results"], 1):
                        st.code(
                            f"[{j}] fused={r['fused_score']:.4f}  "
                            f"sem={r['semantic_score']:.4f}  "
                            f"ent={r['entity_score']:.4f}  "
                            f"bm25={r['bm25_score']:.4f}  "
                            f"temp={r['temporal_score']:.4f}",
                            language=None,
                        )
                        st.caption(r["content"])
                else:
                    st.caption("No memories retrieved.")

        # Save messages to conversation
        chat_messages.append({"role": "user", "content": prompt})
        chat_messages.append({
            "role": "assistant",
            "content": response_text,
            "retrieval": retrieval_info,
        })

        if convo_data.get("title") == "New conversation":
            convo_data["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

        convo_data["messages"] = chat_messages
        save_conversation(username, convo_id, convo_data)


if __name__ == "__main__":
    main()
