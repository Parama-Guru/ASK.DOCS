import streamlit as st
from QAWithPDF.data_indigestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.mode_api import load_model

def main():
    st.set_page_config("QA with Documents")
    st.title('🦜🔗 Welcome to the ChatBot')
    with st.form('my_form'):
        user_question= st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
        submitted = st.form_submit_button('Submit')
    if submitted :
        with st.spinner("Processing..."):
            doc=load_data()
            model=load_model()
            query_engine=download_gemini_embedding(model,doc)
            response=query_engine.query(user_question)
            st.write(response.response)
    
if __name__=="__main__":
    main()