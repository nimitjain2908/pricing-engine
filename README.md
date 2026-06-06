# FinSight Pricing Recommendation Engine

A RAG-powered pricing intelligence tool that answers natural language pricing questions using internal documents, FAISS vector search, and Llama 3.3 70B via Groq.

---

## Demo

Ask questions like:
- *"What should we price our Growth plan at?"*
- *"What discount should we offer for annual plans?"*
- *"How does our pricing compare to competitors?"*
- *"What is the churn rate for monthly vs annual customers?"*

---

## How It Works
User question
↓
Convert to vector embedding (sentence-transformers)
↓
FAISS searches 67 document chunks for most relevant context
↓
Retrieved chunks + question sent to Llama 3.3 70B
↓
Grounded answer returned with source citations

---

## Project Structure
pricing-engine/
├── data/
│   └── pricing_docs/          # 10 internal pricing documents
├── src/
│   ├── ingest.py              # Load docs, chunk, embed, save to FAISS
│   ├── retriever.py           # Semantic search over vectorstore
│   ├── chain.py               # LangChain RAG chain with Groq LLM
│   └── api.py                 # FastAPI REST endpoints
├── app.py                     # Streamlit frontend
└── requirements.txt

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | RAG pipeline orchestration |
| FAISS | Local vector database for semantic search |
| sentence-transformers | Text embedding model (all-MiniLM-L6-v2) |
| Groq + Llama 3.3 70B | LLM for answer generation |
| FastAPI | REST API backend |
| Streamlit | Interactive frontend |
| Pydantic | Request/response validation |

---

## Setup

**1. Clone the repository**
git clone https://github.com/nimitjain2908/pricing-engine.git
cd pricing-engine

**2. Install dependencies**
pip install langchain langchain-community langchain-groq langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers streamlit fastapi uvicorn python-dotenv groq

**3. Add your Groq API key**
GROQ_API_KEY=your_key_here

**4. Build the vectorstore**
python -m src.ingest

**5. Start the API**
python -m src.api

**6. Run the dashboard**
streamlit run app.py

---

## Knowledge Base

The `data/pricing_docs/` folder contains 10 synthetic internal documents covering:

- Competitor pricing analysis
- Internal pricing tiers and rationale
- Market benchmarks by segment
- Discount and exception policy
- Customer segmentation and willingness to pay
- Churn and pricing correlation analysis
- Expansion revenue and upsell strategy
- Geographic pricing adjustments
- Contract terms and billing policy
- Executive pricing recommendations

---

## Built by

Nimit Jain · [LinkedIn](https://linkedin.com/in/nimitjain2908) · [GitHub](https://github.com/nimitjain2908)