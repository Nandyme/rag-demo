import streamlit as st
import os
from rag_utils import RAGPDFProcessor

def main():
    st.title("📄 PDF RAG Interaction App")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("PDF Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF", 
            type=['pdf'], 
            help="Upload a PDF containing images and tables"
        )

    # Main interaction area
    if uploaded_file:
        try:
            # Save uploaded PDF temporarily
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize RAG Processor (now using Streamlit secrets)
            rag_processor = RAGPDFProcessor(
                pdf_path="temp_uploaded.pdf"
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
        st.warning("Please upload a PDF")

if __name__ == "__main__":
    main()
