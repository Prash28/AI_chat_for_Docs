import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient

import os
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Path to Chroma DB
CHROMA_PATH = "chroma_db"

# Prompt template for RAG
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize Hugging Face inference client (online)
client = InferenceClient(api_key=HF_API_TOKEN)

def main():
    parser = argparse.ArgumentParser(description="Query Chroma DB with online LLaMA 3.1 using Hugging Face embeddings")
    parser.add_argument("query_text", type=str, help="Your question or query")
    args = parser.parse_args()
    query_text = args.query_text

    # Load Chroma DB with HuggingFace embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve relevant documents
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    # print("results:")
    # print(results)
    if len(results) == 0 or results[0][1] < 0.2:
        print("Unable to find matching results.")
        return

    # Combine retrieved text into context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Prepare prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("\n--- Prompt sent to model ---\n")
    print(prompt)
    print("\n--- Generating answer ---\n")

    # Query online LLaMA model
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers questions based only on provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
        # stream=False,
        # return_full_text=False
    )

    # Display answer and sources
    answer = response.choices[0].message["content"].strip()
    print("Model Response:\n")
    print(answer)

    sources = [doc.metadata.get("source", None) for doc, _ in results]
    print("\nSources:", sources)


if __name__ == "__main__":
    main()
