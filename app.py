import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FinSight Pricing Engine", layout="wide")
st.title("FinSight Pricing Recommendation Engine")
st.caption("RAG-powered pricing intelligence · Powered by Llama 3.3 70B + FAISS · Built by Nimit Jain")

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("About")
st.sidebar.info("""
This tool uses Retrieval Augmented Generation (RAG) to answer pricing questions 
based on FinSight's internal pricing documents.

**How it works:**
1. Your question is converted to a vector embedding
2. FAISS searches for the most relevant document chunks
3. Llama 3.3 70B generates a grounded answer from those chunks
""")

st.sidebar.divider()
st.sidebar.header("API Status")

try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.status_code == 200:
        st.sidebar.success("API is online")
    else:
        st.sidebar.error("API returned an error")
except:
    st.sidebar.error("API is offline. Run: python -m src.api")

# ── Sample questions ──────────────────────────────────────────────────
st.subheader("Sample questions")
st.caption("Click any question to load it")

try:
    sample_resp = requests.get(f"{API_URL}/sample-questions", timeout=3)
    sample_questions = sample_resp.json()["questions"]
except:
    sample_questions = []

cols = st.columns(2)
for i, q in enumerate(sample_questions):
    if cols[i % 2].button(q, key=f"q_{i}", use_container_width=True):
        st.session_state["selected_question"] = q

st.divider()

# ── Main query interface ──────────────────────────────────────────────
st.subheader("Ask a pricing question")

default_q = st.session_state.get("selected_question", "")
question = st.text_area(
    "Enter your question",
    value=default_q,
    height=100,
    placeholder="e.g. What discount should we offer to win customers from LedgerAI?"
)

col_submit, col_k = st.columns([3, 1])
with col_k:
    k = st.slider("Context chunks", 2, 8, 4,
                  help="Number of document chunks retrieved. More = broader context.")
with col_submit:
    submit = st.button("▶ Get Recommendation", type="primary", use_container_width=True)

# ── Response ──────────────────────────────────────────────────────────
if submit and question.strip():
    with st.spinner("Retrieving context and generating recommendation..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "k": k},
                timeout=30
            )
            result = response.json()

            st.divider()

            st.subheader("Recommendation")
            st.markdown(result["answer"])

            st.divider()

            st.subheader("Sources consulted")
            for source in result["sources"]:
                filename = source.split("\\")[-1].split("/")[-1]
                st.markdown(f"📄 `{filename}`")

        except Exception as e:
            st.error(f"Error: {str(e)}")

elif submit and not question.strip():
    st.warning("Please enter a question first.")

# ── History ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

if submit and question.strip() and "result" in dir():
    st.session_state["history"].append({
        "question": question,
        "answer": result["answer"]
    })

if st.session_state["history"]:
    st.divider()
    with st.expander(f"Question history ({len(st.session_state['history'])} questions)"):
        for i, item in enumerate(reversed(st.session_state["history"])):
            st.markdown(f"**Q{len(st.session_state['history'])-i}: {item['question']}**")
            st.markdown(item["answer"])
            st.divider()