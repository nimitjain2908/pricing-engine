from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.ingest import load_vectorstore

def get_retriever(k=4):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever

def retrieve_context(query: str, k=4):
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
    return context, sources

if __name__ == "__main__":
    query = "What should we price our Growth plan at?"
    context, sources = retrieve_context(query)
    print(f"Query: {query}\n")
    print(f"Sources: {sources}\n")
    print(f"Context:\n{context}")