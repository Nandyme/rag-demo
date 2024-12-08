import streamlit as st
import os
from dotenv import load_dotenv
from rag_utils import RAGPDFProcessor

def main():
    st.title("📄 PDF RAG Interaction App")
    
    # Load environment variables
    load_dotenv()
    
    # Sidebar for PDF upload and API Key
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader(
            "Upload PDF", 
            type=['pdf'], 
            help="Upload a PDF containing images and tables"
        )
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API Key"
        )

    # Main interaction area
    if uploaded_file and openai_api_key:
        # Save uploaded PDF temporarily
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Initialize RAG Processor
            rag_processor = RAGPDFProcessor(
                pdf_path="temp_uploaded.pdf", 
                openai_api_key=openai_api_key
            )
            
            # Query input
            user_query = st.text_input("Ask a question about your PDF")
            
            if user_query:
                with st.spinner('Searching and generating response...'):
                    response = rag_processor.query_document(user_query)
                    st.write("### Response")
                    st.write(response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists("temp_uploaded.pdf"):
                os.remove("temp_uploaded.pdf")
    else:
        st.warning("Please upload a PDF and provide your OpenAI API Key")

if __name__ == "__main__":
    main()
