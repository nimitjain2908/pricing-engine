from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chain import ask
from src.ingest import load_vectorstore
import uvicorn

app = FastAPI(
    title="FinSight Pricing Engine API",
    description="RAG-powered pricing recommendation engine for FinSight",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str
    k: int = 4

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

@app.get("/")
def root():
    return {"status": "ok", "message": "FinSight Pricing Engine is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "vectorstore": "loaded"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        result = ask(request.question)
        return AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample-questions")
def sample_questions():
    return {
        "questions": [
            "What should we price our Growth plan at?",
            "What discount should we offer for annual plans?",
            "How does our pricing compare to competitors?",
            "What is the churn rate for monthly vs annual customers?",
            "What are the recommended pricing changes for Q2 2026?",
            "How should we price for the Indian market?",
            "What is the discount policy for nonprofits?",
            "What add-ons should we launch and at what price?"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)