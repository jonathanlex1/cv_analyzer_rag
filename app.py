from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import streamlit as st 
import os 

os.makedirs('data', exist_ok=True)

def ingestion(data_file_path:str) :
    """
    create vector embeddings from the data
    """
    if not isinstance(data_file_path, str) : 
        raise ValueError('url must be string')
    
    loader = PyPDFLoader(data_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap = 100, separators=['\n'])
    docs_splitted = splitter.split_documents(docs)
    model_embedding = OllamaEmbeddings(model='llama3.2')
    faiss_db = FAISS.from_documents(docs_splitted, model_embedding)
    return faiss_db


def retriever(_retriever) : 
    """
    create retriever chain with prompt, llm and retriever
    """

    prompt_template = """
    ("system", "You are a professional HR assistant helping evaluate how well a CV fits a job description."),
    ("human", 
    "Job Requirement:
    {input}
        
    "
    "Relevant Content from the CV:
    {context}
        
    "
    "Analyze the match and list strengths, gaps, an recommendation.")
    """
    prompt = ChatPromptTemplate([prompt_template])
    llm = OllamaLLM(model = 'llama3.2', temperature=0)
    vector_db = ingestion(_retriever)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    return retriever_chain


def main() : 
    
    st.header('🤖 RAG Chatbot to Analyze and Match Your CV with Job Requirements')

    #upload pdf document
    with st.sidebar:
        st.title("🤖 RAG CV Chatbot")
        st.markdown("### 🎯 Let's Match Your CV to Your Dream Job!")
        st.markdown("Upload your **CV (PDF)** and chat with an AI assistant to check how well you match your desired job.")
        
        uploaded_file = st.file_uploader("📄 Upload your CV here:", type="pdf")

        st.markdown("---")
        st.markdown("💡 *Tip: Make sure your resume is updated for best results!*")
    
        
    # retrieving 
    if uploaded_file : 
        
        uploaded_file_path = os.path.join('data', uploaded_file.name)
            
        with open(uploaded_file_path, 'wb') as file : 
            file.write(uploaded_file.getbuffer())

        query = st.chat_input('Tell your job requirements')

        if query :
            with st.spinner('Wait') : 
                retriever_chain = retriever(uploaded_file_path)
                result = retriever_chain.invoke({'input' : query})
            st.success('✅ Analysis complete!')
            st.write(result['answer'])

        if os.path.exists(uploaded_file_path) : 
            os.remove(uploaded_file_path) 

    else : 
        query = st.chat_input('Tell your job requirements')
        if query : 
            st.warning('You have to upload your cv first')
            
    

if __name__ == '__main__' :
    main()
