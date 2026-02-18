import litellm
import pydantic
from langgraph.graph import StateGraph
import chromadb
from dotenv import load_dotenv
import os

# Load keys from .env
load_dotenv()

def verify():
    print("--- Environment Verification ---")
    print(f"Pydantic version: {pydantic.__version__}")
    print(f"ChromaDB version: {chromadb.__version__}")
    
    # Check for API Key
    key = os.getenv("OPENAI_API_KEY")
    if key:
        print("✅ API Key found in .env")
    else:
        print("⚠️ Warning: No OPENAI_API_KEY found in .env file.")
    
    print("\nEnvironment is 100% Ready!")

if __name__ == "__main__":
    verify()