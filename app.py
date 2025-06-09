from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from pydantic import BaseModel
import os 

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
    return FAISS.from_documents(docs_splitted, model_embedding)

def retriever(vector_db) : 
    """
    create retriever chain with prompt, llm and retriever
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional HR assistant helping evaluate how well a CV fits a job description."),
        ("human", 
         "Job Requirement:\n{input}\n\n"
         "Relevant Content from the CV:\n{context}\n\n"
         "Analyze the match and list strengths, gaps, and recommendations.")
    ])
    llm = OllamaLLM(model = 'llama3.2', temperature=0)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

class Response(BaseModel) : 
    answer:str
    prompt:str

app = FastAPI()

@app.get('/')
def greeting() : 
    return 'hello'

@app.post('/', response_model= Response)
async def upload_file(prompt:str= Form(...), file: UploadFile=File(...)) :
    #upload file
    path = 'data/'
    if not os.path.exists(path) : 
        os.makedirs(path, exist_ok=True)

    content_uploaded = file.filename
    filepath_uploaded = os.path.join(path, content_uploaded)
    
    with open(filepath_uploaded, 'wb') as f : 
        f.write(await file.read())
    
    vector_db = ingestion(filepath_uploaded)

    retriever_chain = retriever(vector_db)

    result = retriever_chain.invoke({'input' : prompt})
    response = result['answer']
    
    return Response(prompt= prompt, answer=response) 


if __name__ == '__main__' :
    uvicorn.run(app, port=8000)
