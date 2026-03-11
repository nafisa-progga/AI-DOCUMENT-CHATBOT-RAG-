# AI Document Chatbot (Advanced RAG System)

## 1. Project Overview
This project is an intelligent chat system developed for high-accuracy document interrogation. It utilizes a **Retrieval-Augmented Generation (RAG)** architecture to provide context-aware answers from internal PDF and DOCX manuals while strictly adhering to hallucination control protocols.


---

## 2. Chosen Architecture
The application follows a decoupled, modular design to ensure high **Architectural Quality** and scalability:

* **Logic Layer (`rag_engine.py`)**: A standalone engine responsible for document ingestion, semantic chunking, vector database management, and managing the conversational retrieval chain.
* **UI Layer (`app.py`)**: A professional **Streamlit** interface handling multi-turn chat sessions, session state, and real-time user feedback.

### Key Technical Components
* **LLM**: **Gemini 1.5 Flash** (Selected for its 1M+ token context window and industry-leading inference speed).
* **Embeddings**: Google Generative AI (`models/text-embedding-004`).
* **Vector Store**: **ChromaDB** (High-performance local vector storage for rapid retrieval).
* **Chunking Strategy**: `RecursiveCharacterTextSplitter` (Chunk Size: 1000, Overlap: 100).
* **Memory**: `ConversationBufferMemory` for persistent conversational context.

---

## 3. Core Features & Implementation
| Feature | Implementation Detail |
| :--- | :--- |
| **Hallucination Control** | Strict system prompt with triple-hash delimiters. Missing info triggers: *"This information is not present in the provided document."* |
| **Conversational Memory** | Tracks chat history for natural follow-up questions. |
| **Full Format Support** | Integrated `PyPDFLoader` and `Docx2txtLoader` for PDF and DOCX processing. |
| **Source Citation** | Expandable "View Source Context" section showing the exact text used for the answer. |
| **Similarity Score** | Displays a **Search Confidence** metric (0-100%) for every retrieval. |
| **Request Logging** | Records all queries and system operations into `app_usage.log` and `app_log.log`. |
| **Prompt Injection Protection** | Advanced instructional guardrails and delimiters (###) to block jailbreak attempts. |
| **Dockerization** | Full `Dockerfile` and `.dockerignore` provided for consistent containerized deployment. |
| **Cloud Ready** | Provided `app.yaml` for seamless deployment to **Google Cloud Platform**. |

---

## 4. Technical Explanation & Justification
* **Why Gemini 1.5 Flash?** It provides the best balance of reasoning and speed. Its massive context window allows for handling significantly larger technical manuals than standard models.
* **Modular Design**: Separating the `RAGEngine` logic from the Streamlit UI ensures that the core AI functionality can be reused in different environments (CLI, API, or Mobile) without rewriting the logic.
* **Vector Embeddings**: Transitioned to `text-embedding-004` to ensure compatibility with the stable Google Generative AI v1 API and improve semantic retrieval accuracy.

---

## 5. Setup & Run Instructions

### Local Setup
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/nafisa-progga/AI-DOCUMENT-CHATBOT-RAG-.git
    cd AI-DOCUMENT-CHATBOT-RAG or <repository-folder>
    ```
2.  **Create & Activate Virtual Environment**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configuration**
    Create a `.env` file in the root directory and add your key:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```
5.  **Run the App**
    ```bash
    streamlit run app.py
    ```

### Docker Setup
```bash
# Build the image
docker build -t csn-chatbot .

# Run the container (Access at http://localhost:8501)
docker run -p 8501:8501 --env-file .env csn-chatbot