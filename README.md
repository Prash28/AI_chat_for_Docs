````markdown
# 🧠 AI Chat for Documents

![Python](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-API-blueviolet)

**AI Chat for Documents** is a **Retrieval-Augmented Generation (RAG)** application that allows you to query your documents and get AI-generated answers. It uses **Hugging Face LLaMA models** for online inference, **Chroma** for vector storage, and **Hugging Face embeddings** for semantic search.

---

## ⚡ Features

- Query `.md`, `.txt`, `.pdf`, and `.json` documents.
- Uses **Chroma** vector database to store embeddings.
- Online inference with Hugging Face **LLaMA 3.1** models.
- Simple CLI interface to ask questions.
- Provides sources for each answer.
- Preloaded example data:
  - Markdown file: `Alice in Wonderland.md`
  - PDF file: `Introduction to AI.pdf`  
  You can replace these files with your own documents to query on new data.

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Prash28/AI_chat_for_Docs.git
cd AI_chat_for_Docs
````

### 2️⃣ Set up a virtual environment

```bash
python -m venv virtualenv
# Windows
virtualenv\Scripts\activate
# macOS/Linux
source virtualenv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Hugging Face API token

Create a `.env` file in the project root:

```bash
HF_API_TOKEN=your_huggingface_api_token
```

---

## 📂 How it Works

### 1️⃣ `createDatabase.py`

* Loads your documents from the `data/` folder (supports `.md`, `.pdf`, `.txt`, `.json`).
* Splits large documents into chunks using `RecursiveCharacterTextSplitter`.
* Generates embeddings for each chunk using Hugging Face `sentence-transformers` models.
* Stores all embeddings in a **Chroma vector database** (`chroma_db`) for fast retrieval.

**Run:**

```bash
python createDatabase.py
```

This creates or refreshes your Chroma database with the current documents.

---

### 2️⃣ `queryDatabase.py`

* Accepts a query via CLI.
* Searches the Chroma database for relevant chunks.
* Prepares a **prompt** using retrieved context.
* Sends the prompt to Hugging Face **LLaMA 3.1** online model for answer generation.
* Displays the answer along with the sources of the retrieved chunks.

**Run:**

```bash
python queryDatabase.py "Who is Alice?"
```

Example output:

```
Model Response:
Alice is the main character in the story...
Sources: ['data/Alice in Wonderland.md']
```

---

## 🗂️ Project Structure

```
AI_chat_for_Docs/
│
├── data/                 # Folder containing documents (.md, .txt, .pdf, .json)
├── chroma_db/            # Generated Chroma vector database
├── createDatabase.py     # Script to create vector database from documents
├── queryDatabase.py      # Script to query database with prompts
├── requirements.txt      # Python dependencies
├── .env                  # Environment file for API tokens (ignored by git)
└── README.md
```

---

## 🔧 Dependencies

* `langchain`
* `langchain-huggingface`
* `langchain-chroma`
* `huggingface_hub`
* `python-dotenv`
* `PyPDF2` (for PDF support)

---

## ⚙️ Configuration

* Adjust **chunk size** and **overlap** in `createDatabase.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100
)
```

* Change the Hugging Face model in `queryDatabase.py` if needed:

```python
model="meta-llama/Llama-3.1-8B-Instruct"
```

---

## 💡 Notes

* This project uses **online inference**. No model download is required.
* Ensure `.env` is never committed to version control.
* Works best with clear, concise prompts.

