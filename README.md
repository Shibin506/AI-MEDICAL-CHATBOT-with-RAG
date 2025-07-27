# 🩺 MediBot: AI Medical Chatbot using RAG

## Project Overview

MediBot is an intelligent medical chatbot designed to provide answers to health-related questions by referencing a curated collection of medical PDF documents. Unlike traditional chatbots that rely solely on their pre-trained knowledge, MediBot utilizes **Retrieval Augmented Generation (RAG)** to ensure its responses are accurate, factual, and directly supported by the provided medical texts.

**Think of it like this:** Instead of making up answers, MediBot "looks up" the information in its digital library of medical textbooks (your PDFs) and then "explains" what it found.

## How it Works: The RAG Pipeline (Modular 3 Phases)

The project is structured into three distinct and modular phases, mirroring the RAG process:

### Phase 1: Building the Brain & Memory (Vector Database)

This phase prepares your custom knowledge base, transforming raw PDF documents into a highly searchable format for the chatbot.

1.  **Load Raw PDF(s):**
    * **What:** The system reads and extracts all text content from medical PDF files stored in the `data/` directory.
    * **Why:** Raw PDFs are like images of text to a computer. We need to get the actual words out.
    * **Tools:** `PyPDFLoader` and `DirectoryLoader` (from `langchain-community`) are used for efficient text extraction from multiple PDFs.

2.  **Create Chunks:**
    * **What:** The extracted text is broken down into smaller, manageable pieces (chunks), typically around 500 characters long with a 50-character overlap between them.
    * **Why:** Large Language Models (LLMs) have a limited "short-term memory" (context window). Breaking text into chunks ensures the LLM can process relevant information efficiently without being overwhelmed.
    * **Tool:** `RecursiveCharacterTextSplitter` (from `langchain`) intelligently slices the text while preserving context.

3.  **Create Vector Embeddings:**
    * **What:** Each text chunk is converted into a unique numerical "fingerprint" or "secret code" called a vector embedding. Text chunks with similar meanings will have similar numerical representations.
    * **Why:** Computers understand numbers better than words. These numerical codes allow for rapid and accurate semantic (meaning-based) searches.
    * **Tool:** `HuggingFaceEmbeddings` (from `langchain-huggingface`) with the `sentence-transformers/all-MiniLM-L6-v2` model is used for this conversion. This model is known for its effectiveness in generating high-quality embeddings.

4.  **Store Embeddings in FAISS:**
    * **What:** All the generated vector embeddings are stored in a specialized, super-fast database called FAISS (Facebook AI Similarity Search).
    * **Why:** FAISS is optimized for quickly finding the "closest" (most similar) numerical fingerprints to a given query, making information retrieval instantaneous. This FAISS database serves as MediBot's "memory."

### Phase 2: Connecting Memory to the Talker (LLM Integration)

This phase links the prepared medical memory to a powerful AI model that can understand your questions and formulate answers.

1.  **Setup Large Language Model (LLM):**
    * **What:** A powerful AI model capable of understanding human language and generating coherent text is selected as the chatbot's "talker" or "brain."
    * **Why:** This model provides the core intelligence for language comprehension and generation.
    * **Tools:** While originally designed for `HuggingFaceEndpoint` (using models like `Mistral-7B-Instruct-v0.3`), the project currently defaults to `ChatGroq` (using models like `meta-llama/llama-4-maverick-17b-128e-instruct`). `ChatGroq` is chosen for its exceptional speed and stable API integration during development.

2.  **Retrieval Augmented Generation (RAG) Chain:**
    * **What:** When you ask a question, the system performs a two-step process:
        1.  Your question is also converted into a vector embedding, which is then used to quickly search the FAISS database for the top 3 most relevant text chunks from your medical PDFs.
        2.  These retrieved text chunks (the "context") are then combined with your original question and fed to the LLM. The LLM is instructed to generate an answer *only* based on this provided context. If the answer isn't in the context, it's instructed to say "I don't know."
    * **Why:** This ensures accuracy, reduces "hallucinations" (AI making up facts), and provides verifiable answers.
    * **Tool:** `RetrievalQA` (from `langchain`) orchestrates this entire process. A `PromptTemplate` is used to guide the LLM's behavior, ensuring it adheres strictly to the provided context.

### Phase 3: Building the Face for Interaction (Chatbot UI)

This final phase creates an intuitive and user-friendly web interface for interacting with MediBot.

1.  **Chatbot with Streamlit:**
    * **What:** A dynamic and interactive chat window is built as a web application.
    * **Why:** Streamlit allows for rapid development of beautiful web UIs using pure Python, making the chatbot accessible without complex web development knowledge.
    * **Tool:** `Streamlit` is the primary framework for the user interface.

2.  **Load Vector Store in Cache:**
    * **What:** The FAISS vector database (MediBot's memory) is loaded once when the Streamlit application starts and kept readily available.
    * **Why:** This significantly improves performance. Loading the large database only once prevents delays during subsequent user interactions, making the chatbot feel fast and responsive.
    * **Tool:** Streamlit's `@st.cache_resource` decorator automatically handles this caching.

## Technologies Used

* **Python:** The core programming language for the entire project, chosen for its extensive ecosystem of AI/ML libraries.
* **LangChain:** A powerful framework for developing applications powered by LLMs, simplifying the integration of various AI components.
* **Hugging Face Ecosystem:**
    * **`HuggingFaceEmbeddings`:** For generating text embeddings.
    * **`HuggingFaceEndpoint`:** (Optional) For connecting to Hugging Face's hosted LLMs.
    * **`sentence-transformers/all-MiniLM-L6-v2`:** The specific embedding model used.
* **FAISS (Facebook AI Similarity Search):** A high-performance library for efficient similarity search of dense vectors, serving as our vector database.
* **Groq:** A specialized platform providing incredibly fast LLM inference, currently used for the chatbot's main LLM (`meta-llama/llama-4-maverick-17b-128e-instruct`).
* **Streamlit:** A Python library for quickly building and deploying interactive web applications, used for the chatbot's user interface.
* **`python-dotenv`:** For securely loading environment variables (like API keys) from a `.env` file.
* **VS Code:** The Integrated Development Environment (IDE) used for coding.

## Project Setup & Running MediBot

Follow these steps to get MediBot up and running on your local machine:

### 1. Prerequisites

* **Python 3.12:** The project requires Python version 3.12. Ensure it's installed and added to your system's PATH.
* **Pipenv:** Install Pipenv globally: `pip install pipenv`
* **API Keys:**
    * **Hugging Face Token:** Obtain a `read` token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    * **Groq API Key:** Obtain an API key from [console.groq.com/keys](https://console.groq.com/keys).
* **PDF Data:** Place your medical PDF documents inside a folder named `data/` in your project's root directory.

### 2. Project Initialization

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/ShibinS06/AI-MEDICAL-CHATBOT-with-RAG.git](https://github.com/ShibinS06/AI-MEDICAL-CHATBOT-with-RAG.git) MediBot_Project
    cd MediBot_Project
    ```
    *(If you're continuing from a local directory where you pushed the code, just `cd` into it.)*

2.  **Create `.env` File:**
    * Create a file named `.env` in your project's root directory.
    * Add your API keys to it:
        ```
        HF_TOKEN="your_huggingface_token_here"
        GROQ_API_KEY="your_groq_api_key_here"
        ```

3.  **Install Dependencies:**
    * Run Pipenv to install all required libraries. If you have multiple Python versions, explicitly specify Python 3.12:
        ```bash
        pipenv install --python "C:\Users\YourUsername\AppData\Local\Programs\Python\Python312\python.exe"
        ```
        (Replace `YourUsername` and ensure the path is correct for your system.)

### 3. Build Chatbot Memory (Phase 1)

This step creates the FAISS vector database from your PDFs.

```bash
pipenv run python create_memory_for_llm.py
