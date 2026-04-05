import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def create_qa_chain(csv_path, groq_api_key):
    # Load CSV 
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()

    # Split CSV rows into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    texts = text_splitter.split_documents(documents)

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='llama-3.1-8b-instant'
    )

    # Same prompt as PDF version
    template = (
        "You are an expert assistant. Use ONLY the provided context to answer.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    )
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            'context': retriever | (
                lambda docs: '\n\n'.join([d.page_content for d in docs])
            ),
            'question': RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def chat_with_csv(csv_file, user_input):
    groq_api_key = os.getenv('GROQ_API_KEY')

    if not groq_api_key:
        return '❌ GROQ_API_KEY not found. Please add it to your environment variables.'

    if not csv_file:
        return 'Please upload a CSV file first.'

    try:
        csv_path = csv_file.name  # file saved by FastAPI
        rag_chain = create_qa_chain(csv_path, groq_api_key)
        response = rag_chain.invoke(user_input)
        return response

    except Exception as e:
        return f'⚠️ Error: {e}'
