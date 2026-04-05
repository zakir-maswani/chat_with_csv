from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain_groq.chat_models import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.outputs import StrOutputParser
import os

def create_qa_chain(pdf_path, groq_api_key):
    """ Load, split, embed and create a custom QA chain."""

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    document_splitters = RecursiveCharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap= 150
    )
    
    texts = document_splitters.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name= ""
    )

    vectorstore = faiss.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs= {"k": 3})

    llm = ChatGroq(
        groq_api_key= groq_api_key,
        model_name = ""
    )

    template = """
    You are an expert assistant. Use ONLY the provided context to answer or summarize.
    If the answer in the context is not, say I don't have enough information.

    Context:
    {context}

    Question:
    {question}

    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever | (
                lambda docs: "\n\n".join([d.page_content for d in docs])
            ),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser
    )

    return chain


def chat_with_pdf(pdf_file, user_query):
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        return "GROQ_API_KEY not found."
    
    if not pdf_file:
        return "Please upload a PDF file first"
    
    try:
        pdf_path = pdf_file.name
        rag_chain = create_qa_chain(pdf_path, groq_api_key)
        response = rag_chain.invoke(user_query)
        return response
    except Exception as e:
        return f"Error: {e}"