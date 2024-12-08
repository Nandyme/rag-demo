import os
import streamlit as st
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

class RAGPDFProcessor:
    def __init__(self, pdf_path: str):
        # Use Streamlit secrets to get API key
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        self.pdf_path = pdf_path
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._process_pdf()
        self.qa_chain = self._create_qa_chain()

    def _process_pdf(self):
        # Load PDF and split into chunks
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings
        )
        return vectorstore

    def _create_qa_chain(self):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        return qa_chain

    def query_document(self, query: str) -> str:
        return self.qa_chain.run(query)
