# connect_memory_with_llm.py content (UPDATED)
import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq # ADD THIS IMPORT

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

print("DEBUG: Environment variables loaded.")


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# NOTE: This load_llm function will no longer be used for Mistral in this script
# It's kept here as a placeholder to avoid errors if other parts try to call it.
def load_llm(huggingface_repo_id):
    pass # This function will do nothing for now in this script
    # The Groq LLM will be instantiated directly in the qa_chain below
    # This avoids the HuggingFaceEndpoint issues for the command-line test

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(f"DEBUG: Loading FAISS database from: {DB_FAISS_PATH}")
try:
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("DEBUG: FAISS database loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load FAISS database: {e}")
    print("Please ensure 'create_memory_for_llm.py' ran successfully and created the 'vectorstore/db_faiss' directory.")
    raise


# Create QA chain
print("DEBUG: Creating QA chain...")
try:
    # Use Groq LLM directly for this test script
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

    llm_for_qa = ChatGroq(
        model_name="llama3-8b-8192", # A commonly available and free-tier friendly Groq model
        temperature=0.0,
        groq_api_key=GROQ_API_KEY,
    )
    print("DEBUG: Using ChatGroq LLM for QA chain.")

    qa_chain=RetrievalQA.from_chain_type(
        llm=llm_for_qa, # Use the Groq LLM
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    print("DEBUG: QA chain created successfully.")
except Exception as e:
    print(f"ERROR: Failed to create QA chain: {e}")
    raise


# Now invoke with a single query
print("DEBUG: Prompting user for query...")
user_query=input("Write Query Here: ")
print(f"DEBUG: User query received: {user_query}")

print("DEBUG: Invoking QA chain...")
try:
    response=qa_chain.invoke({'query': user_query})
    print("DEBUG: QA chain invoked successfully.")
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])
except Exception as e:
    print(f"ERROR: Failed to invoke QA chain: {e}")
    print("Possible reasons: LLM connectivity, invalid API key, or issue during retrieval.")
    raise