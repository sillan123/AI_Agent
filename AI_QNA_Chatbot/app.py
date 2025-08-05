import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

PDF_FOLDER = "pdfs"
VECTOR_DB_PATH = "vector_store"
EMBED_MODEL = "all-MiniLM-L6-v2"

# -------------- Load & Split PDFs ---------------
@st.cache_resource
def load_and_split_documents():
    all_docs = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
            pages = loader.load()
            all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)
    return split_docs

# -------------- Build or Load Vector Store -------
# @st.cache_resource
# def get_vectorstore(_documents):
#     embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     if os.path.exists(VECTOR_DB_PATH):
#         return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
#     else:
#         vectorstore = FAISS.from_documents(_documents, embeddings)
#         vectorstore.save_local(VECTOR_DB_PATH)
#         return vectorstore

@st.cache_resource
def get_vectorstore(_documents):
    embeddings = HuggingFaceEmbeddings()

    faiss_index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")

    if os.path.exists(faiss_index_path):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(_documents, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        return vectorstore

# -------------- Initialize LLM --------------------
@st.cache_resource
def get_llm():
    return Ollama(model="mistral")  # Make sure `ollama run mistral` is working

# -------------- Build QA Chain --------------------
@st.cache_resource
def get_qa_chain():
    docs = load_and_split_documents()
    vectorstore = get_vectorstore(docs)
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# -------------- Streamlit UI ---------------------
st.set_page_config(page_title="HR Chatbot", page_icon="🤖")
st.title("🤖 HR Chatbot")
st.markdown("Ask your HR-related queries based on internal policy documents.")

query = st.text_input("Ask a question:")
if query:
    with st.spinner("Thinking..."):
        qa_chain = get_qa_chain()
        result = qa_chain(query)
        st.success(result["result"])

        with st.expander("See source document chunks"):
            for doc in result['source_documents']:
                st.write(doc.page_content[:300] + "...")
