from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.retriever import retrieve_context
from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """You are an expert pricing strategist for a B2B SaaS fintech company called FinSight.
You answer pricing questions based strictly on the internal documents provided as context.
Always cite specific numbers, percentages, and recommendations from the context.
If the context does not contain enough information to answer the question, say so clearly.
Do not make up pricing figures or recommendations not present in the context.
Keep answers concise, structured, and actionable."""

def build_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1000,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
{system_prompt}

CONTEXT FROM INTERNAL DOCUMENTS:
{context}

SOURCES: {sources}

QUESTION: {question}

ANSWER:""")

    chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask(question: str):
    context, sources = retrieve_context(question)
    chain = build_chain()

    response = chain.invoke({
        "system_prompt": SYSTEM_PROMPT,
        "context": context,
        "sources": ", ".join(sources),
        "question": question
    })

    return {
        "question": question,
        "answer": response,
        "sources": sources
    }

if __name__ == "__main__":
    result = ask("What discount should we offer for annual plans?")
    print(f"Question: {result['question']}\n")
    print(f"Answer: {result['answer']}\n")
    print(f"Sources: {result['sources']}")