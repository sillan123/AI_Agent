import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

PDF_FOLDER = "pdfs"
VECTOR_DB_PATH = "vector_store"

os.makedirs(PDF_FOLDER, exist_ok=True)

# ----------------- Cache Functions -----------------
@st.cache_resource
def load_and_split_documents(uploaded_files=None):
    all_docs = []

    # Load PDFs from static folder
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
            all_docs.extend(loader.load())

    # Handle newly uploaded files
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(PDF_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(all_docs)

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

@st.cache_resource
def get_llm():
    return Ollama(model="mistral")

def build_qa_chain(documents):
    vectorstore = get_vectorstore(documents)
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="AI/ML/GENAI QnA Chatbot", page_icon="🤖")
st.title("🤖 AI/ML/GENAI QnA Chatbot")
st.markdown("Ask your AI/ML questions. This chatbot answers from uploaded and existing PDFs.")

# File uploader
uploaded_files = st.file_uploader("📎 Upload PDFs (Optional):", type=["pdf"], accept_multiple_files=True)

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Question Input
user_input = st.text_input("💬 Ask a question:")

# ----------------- Main QA Logic -----------------
if user_input:
    with st.spinner("Thinking..."):
        documents = load_and_split_documents(uploaded_files)
        qa_chain = build_qa_chain(documents)
        result = qa_chain(user_input)

        answer = result["result"]
        sources = result.get("source_documents", [])

        # Save in session
        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer,
            "sources": sources,
            "feedback": None  
        })

        st.success(answer)

        # Show source chunks
        with st.expander("📄 Source Chunks"):
            for doc in sources:
                st.write(doc.page_content[:300] + "...")

# ----------------- Chat History with Feedback -----------------
if st.session_state.chat_history:
    st.markdown("### 📝 Chat History")
    for idx, entry in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**{idx}. You:** {entry['question']}")
        st.markdown(f"**🤖 Answer:** {entry['answer']}")

        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            if st.toggle(f"👍 Helpful (Q{idx})", key=f"up_{idx}"):
                entry["feedback"] = "👍"
        with feedback_col2:
            if st.toggle(f"👎 Not Helpful (Q{idx})", key=f"down_{idx}"):
                entry["feedback"] = "👎"

        # Optionally show feedback status
        if entry["feedback"]:
            st.markdown(f"**Feedback received:** {entry['feedback']}")
