import streamlit as st
from QAWithPDF.data_indigestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.mode_api import load_model
import os
import shutil
import atexit

def main():
    
    save_dir = 'data'
    vector_dir = 'storage'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def cleanup():
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
    atexit.register(cleanup)

    def cleanup():
        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir)
    atexit.register(cleanup)

    st.set_page_config(page_title="QA with Documents")

    doc_file = st.file_uploader("Upload your document", type=['txt', 'pdf'])

    if doc_file is not None:
        file_path = os.path.join(save_dir, doc_file.name)
        with open(file_path, 'wb') as f:
            f.write(doc_file.getbuffer())
        
        st.success(f"File {doc_file.name} uploaded successfully and saved at {file_path}.")

    st.header("QA Chatbot")

    user_question = st.text_input("Ask your question")

    if st.button("Submit & Process"):
        if doc_file is not None and user_question:
            with st.spinner("Processing..."):
                doc = load_data()
                model = load_model()
                query_engine = download_gemini_embedding(model, doc)
                response = query_engine.query(user_question)
                st.write(response.response)
        else:
            st.error("Please upload a document and enter a question.")

if __name__ == "__main__":
    main()