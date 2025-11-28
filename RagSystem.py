# RagSystem.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os  # ✅ NEW

# ✅ This will load .env locally (on your laptop).
# On Streamlit Cloud, it will just do nothing (which is fine).
load_dotenv()


def Setup_Vector_Store(text: str) -> FAISS:
    """Create and return a FAISS vector store from transcript text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def Use_Rag_System(vector_store: FAISS, question: str) -> str:
    """Use existing vector store to answer a question."""

    # ✅ Get HF token from environment (works for .env + Streamlit Secrets)
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN is not set. "
            "Set it in your .env (locally) and in Streamlit Secrets on the cloud."
        )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # ✅ Pass the token explicitly (optional but safer)
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        max_new_tokens=256,
        huggingfacehub_api_token=hf_token,  # ✅ NEW
    )
    model = ChatHuggingFace(llm=llm, temperature=0.0)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser

    result = main_chain.invoke(question)
    return result
