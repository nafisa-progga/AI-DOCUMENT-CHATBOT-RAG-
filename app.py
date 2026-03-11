import streamlit as st
import tempfile
import os
import logging
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_usage.log"),
        logging.StreamHandler()
    ]
)


st.set_page_config(page_title="XYZ Company AI Chatbot", layout="wide")
st.title(" AI Document Chatbot (RAG System)")

if not GEMINI_API_KEY:
    st.error(" GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []


st.sidebar.header("Document Management")
uploaded_file = st.sidebar.file_uploader("Upload Internal Document", type=["pdf", "docx"])

if uploaded_file:
    # Initialize RAG Engine only once per upload
    if "qa_chain" not in st.session_state:
        with st.spinner("Processing document and generating embeddings..."):
            try:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Initialize Engine and Chain
                engine = RAGEngine(tmp_path, GEMINI_API_KEY)
                st.session_state.vector_db = engine.process_document()
                st.session_state.qa_chain = engine.get_qa_chain(st.session_state.vector_db)
                
                st.sidebar.success(f" {uploaded_file.name} loaded successfully!")
                logging.info(f"Successfully processed file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing document: {e}")
                logging.error(f"Document processing failed: {e}")
                st.stop()

    # Chat Interface - display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about the document:"):
        # Log and display user message
        logging.info(f"User Query: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                # Similarity Score - search the vector DB for the closest match to show confidence
                docs_with_scores = st.session_state.vector_db.similarity_search_with_relevance_scores(prompt, k=1)
                
                # Get Answer from the Conversational Chain
                response = st.session_state.qa_chain({"question": prompt})
                answer = response['answer']
                st.markdown(answer)
                
                # METRICS & CITATIONS 
                if answer != "This information is not present in the provided document.":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if docs_with_scores:
                            # score is usually between 0-1
                            raw_score = docs_with_scores[0][1]
                            # Handle potential negative scores or outliers depending on distance metric
                            display_score = max(0, min(100, raw_score * 100))
                            st.metric("Search Confidence", f"{round(display_score, 2)}%")
                    
                    with col2:
                        with st.expander("View Source Context"):
                            for i, doc in enumerate(response.get('source_documents', [])):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(f"{doc.page_content[:250]}...")
                                st.divider()
                
                logging.info(f"AI successful response for: {prompt}")

            except Exception as e:
                answer = "An error occurred while generating the response."
                st.error(f"Error: {e}")
                logging.error(f"Chain execution error: {e}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info(" Welcome! Please upload a PDF or DOCX file in the sidebar to start chatting.")
