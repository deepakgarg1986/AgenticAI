import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ✅ System Prompt for LLM
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. 
Do not include any external knowledge. Clearly state if the context doesn't provide enough information.
"""

# ✅ Function to Process Multiple PDFs
def process_documents(uploaded_files: list[UploadedFile]) -> list[Document]:
    """Processes multiple uploaded PDF files by extracting text and chunking it."""
    all_docs = []

    for uploaded_file in uploaded_files:
        try:
            # Store uploaded file as a temp file
            with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Load and split the document
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from {uploaded_file.name}")
            # Delete the temp file after processing
            try:
                os.unlink(temp_file_path)
            except PermissionError:
                st.warning(f"Could not delete temporary file: {temp_file_path}")

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400000,
                chunk_overlap=100,
                #separators=["\n\n", "\n", ".", "?", "!", " ", ""],
                separators=[]
            )
            all_docs.extend(text_splitter.split_documents(docs))

        except Exception as e:
            st.error(f"Error processing document {uploaded_file.name}: {e}")

    return all_docs


# ✅ Function to Get or Create Vector Collection
def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    #print("I am get vector collection")
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    #chroma_client.de
    #collections = chroma_client.get_or_create_collection(
    #    name="rag_app",
    #    embedding_function=ollama_ef,
    #    metadata={"hnsw:space": "cosine"},
    #)
    
    #collections.delete(ids=["SlowHTTPRequests_18"])
    #for collection in chroma_client.list_collections():
     #   print(collection.delete)
     #   chroma_client.delete_collection(collection.name)

    # Get all collections
    #print("I have created client")
    
    #collections = chroma_client.list_collections()
    #print(collections)
 
# Loop through each collection and print all data
    #for col_meta in collections:
    #    collection = chroma_client.get_collection(name=col_meta.name)
    #    print(f"Collection: {col_meta.name}")
        #print("Collection:" + collection)
    #results = collection.get(include=["documents", "metadatas", "embeddings"])
    #print(results)
    #for i, doc in enumerate(results["documents"]):
    #    print(f"  ID: {results['ids'][i]}")
    #    print(f"  Document: {doc}")
    #    print(f"  Metadata: {results['metadatas'][i]}")
    #    print()    
    #col = chroma_client.get_or_create_collection(name="rag_app",embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"})
    #print(col.get(include=["documents", "metadatas", "embeddings"]))
    #print(col.query(query_texts=["SlowHTTPRequests_1"], n_results=1))
    #print(col.get(where={'page':{'$eq': 4}}))
    return chroma_client.get_or_create_collection(name="rag_app",embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"})
    #return col.get(include=["documents", "metadatas", "embeddings"])
    #return chroma_client.get_or_create_collection(
    #    name="rag_app",
    #    embedding_function=ollama_ef,
    #    metadata={"hnsw:space": "cosine"},
    #)


# ✅ Function to Add Multiple Documents to ChromaDB
def add_to_vector_collection(all_splits: list[Document]):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        #metadatas.append(split.metadata)
        metadatas.append({"TSG": 'RIUtilization'})
        ids.append(f"RIUtilization_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("All documents have been added to the vector store!")


# ✅ Function to Query ChromaDB
def query_collection(prompt: str, n_results: int = 15):
    print(prompt)
    if not prompt.strip():
        raise ValueError("Query prompt cannot be empty.")
    
    collection = get_vector_collection()
    #print(collection.get(where={'page':{'$eq': 4}}))
    results = collection.get(where={'TSG':{'$eq': 'RIUtilization'}})
    #results= collection.query()
    #results = collection.query(query_texts=[prompt], n_results=n_results)
    #print(results)
    return results


# ✅ Function to Call LLM for Answer Generation
def call_llm(context: str, prompt: str):
    try:
        response = ollama.chat(
            model="llama2",
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
            ],
        )
        for chunk in response:
            if not chunk["done"]:
                yield chunk["message"]["content"]
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        yield "An error occurred while generating the response."


# ✅ Function to Re-Rank Retrieved Documents
def re_rank_cross_encoders(documents: list[str], prompt: str) -> tuple[str, list[int]]:
    try:
        if not documents:
            raise ValueError("No documents provided for re-ranking.")

        encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranks = encoder_model.rank(prompt, documents, top_k=6)

        relevant_text = ""
        relevant_text_ids = []
        for rank in ranks:
            relevant_text += documents[rank["corpus_id"]] + "\n\n"
            relevant_text_ids.append(rank["corpus_id"])

        return relevant_text, relevant_text_ids
    except Exception as e:
        st.error(f"Error during document re-ranking: {e}")
        return "", []


# ✅ Streamlit UI
if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title="RAG-Based Multi-PDF Q&A")

        # ✅ Upload multiple PDFs
        uploaded_files = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=True
        )

        process = st.button("⚡️ Process PDFs")
        if uploaded_files and process:
            st.info(f"Processing {len(uploaded_files)} PDFs...")

            all_splits = process_documents(uploaded_files)
            if all_splits:
                add_to_vector_collection(all_splits)
            else:
                st.error("Failed to process the documents.")

    # ✅ Q&A Section
    st.header("🗣️ RAG-Based Q&A System")
    prompt = st.text_area("**Ask a question related to your documents:**")
    ask = st.button("🔥 Ask")
    output = ""
    if ask and prompt:
        results = query_collection(prompt)
        print("Number of documents retrieved:", len(results.get("documents")))
        if results and results.get("documents"):
            #context = results.get("documents")[0]
            context = results.get("documents")[0:]
            for rank in range(len(results.get("documents")) ):
                print(rank)
                output += results.get("documents")[rank] + "\n\n"
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context, prompt)
            print(output)
            if relevant_text:
                #response = call_llm(context=relevant_text, prompt=prompt)
                #st.write_stream(response)
                #st.write_stream(relevant_text)
                #st.write_stream(context)
                #st.write(context)
                st.write(output)


                with st.expander("See retrieved documents"):
                    st.write(results)

                with st.expander("See most relevant document IDs"):
                    st.write(relevant_text_ids)
                    st.write(relevant_text)
            else:
                st.error("No relevant documents found.")
        else:
            st.error("No results found for the query.")
