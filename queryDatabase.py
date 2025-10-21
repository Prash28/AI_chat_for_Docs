import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Paths and constants
CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize Hugging Face inference client
client = InferenceClient(api_key=HF_API_TOKEN)


def query_database(query_text: str) -> str:
    """Query the Chroma DB and return the model's response as a string."""

    # Load Chroma DB with Hugging Face embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve relevant documents
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.2:
        return "⚠️ Unable to find matching results in the database."

    # Combine retrieved text into context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Prepare prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query online LLaMA model
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers questions based only on provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )

    # Extract model answer
    answer = response.choices[0].message["content"].strip()
    return answer


def main(query_text: str = None):
    """Main entry point for CLI or function call."""
    if query_text is None:
        # CLI mode
        parser = argparse.ArgumentParser(description="Query Chroma DB with online LLaMA 3.1")
        parser.add_argument("query_text", type=str, help="Your question or query")
        args = parser.parse_args()
        query_text = args.query_text

    # Run the query
    answer = query_database(query_text)

    # When run via CLI, print the answer
    if __name__ == "__main__":
        print("Model Response:\n")
        print(answer)

    return answer


if __name__ == "__main__":
    main()
