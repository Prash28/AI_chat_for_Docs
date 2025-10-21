import streamlit as st
import os
from createDatabase import save_to_chroma, load_documents, split_text
from queryDatabase import main as query_main  # Make sure this returns a string (the answer)

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# Page configuration
st.set_page_config(page_title="AI Chat for Docs", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Document", "Ask Question"])

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("📄 AI Chat for Documents")
    st.write(
        "Welcome! This app lets you upload your documents and ask AI-powered questions "
        "based on their content. You can upload PDFs, text files, or Markdown files."
    )

# -------------------------------
# Upload Document Page
# -------------------------------
elif page == "Upload Document":
    st.title("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a file", type=["md", "txt", "pdf", "json"])
    
    if uploaded_file:
        # Save uploaded file to data folder
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Saved file: {uploaded_file.name}")

        # Process document and create Chroma database
        with st.spinner("Processing document and creating database..."):
            documents = load_documents()
            chunks = split_text(documents)
            save_to_chroma(chunks)
        
        st.success("🎉 Database created successfully!")

# -------------------------------
# Ask Question Page
# -------------------------------
elif page == "Ask Question":
    st.title("❓ Ask a Question")
    query_text = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        if not query_text.strip():
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Get the result from queryDatabase.main()
                    answer = query_main(query_text)
                    if answer:
                        st.subheader("💬 Answer:")
                        st.write(answer)
                    else:
                        st.info("No relevant information found in the database.")
                except Exception as e:
                    st.error(f"⚠️ Error while processing your question: {e}")
