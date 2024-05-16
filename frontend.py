import streamlit as st
from io import StringIO
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os


class Element(BaseModel):
    type: str
    text: Any

#embeddings model setup -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
model_name="BAAI/bge-large-en"
model_kwargs={'device':'cuda'}
embeddings=HuggingFaceBgeEmbeddings(model_name=model_name,model_kwargs=model_kwargs)
print("Embedding model loaded")
#embeddings model setup -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#qdrant setup -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
url="http://localhost:6333"
collection_name="test_collection"
client=QdrantClient(url,prefer_grpc=False)
print("Qdrant client created")
qdrant=Qdrant(client=client,collection_name=collection_name,embeddings=embeddings)
print("Qdrant object created")
retriever = qdrant.as_retriever()
print("Qdrant object converted to retriever")
#qdrant setup -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# upload pdf file -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if os.path.isfile(f"pdfs/{uploaded_file.name}"):
        continue
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
#writing pdf to pdfs folder
for uploaded_file in uploaded_files:
    with open(f"pdfs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write(f"Saved {uploaded_file.name} wait for processing...")
    # saving embeddings to qdrant
    output_dir = "images"

    # Get texts
    loader=PyPDFLoader(f"pdfs/{uploaded_file.name}")
    documents=loader.load()
    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=f"pdfs/{uploaded_file.name}",
        # Using pdf format to find embedded image blocks
        extract_images_in_pdf=True,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        # Hard max on chunks
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=output_dir,
    )
    st.write("PDF partitioned, now categorizing elements...")
    # Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    print(len(table_elements))
    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOllama(model="llama3")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Apply to tables
    tables = [i.text for i in table_elements]
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    table_summaries = " ".join(table_summaries)
    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
    )

    text_chunks=text_splitter.split_documents(documents)

    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
    )

    table_chunks=text_splitter.split_text(table_summaries)

    st.write("Text and tables split, now saving to Qdrant...")

    qdrant=Qdrant.from_documents(
    text_chunks,
    embeddings,
    url=url,
    collection_name=collection_name,
    prefer_grpc=False
    )

    print("text index in qdrant")

    qdrant=Qdrant.from_texts(
    table_chunks,
    embeddings,
    url=url,
    collection_name=collection_name,
    prefer_grpc=False
    )

    print("tables index in qdrant")
    st.write("Text and tables saved to Qdrant, ready for retrieval!")
#upload pdf file -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# chain setup -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Prompt template
template = """Answer the question based only on the following context, 
which can include text and tables,
if answer is present in tables, please provide in good format,
if the question has anything unrelated please ask a probing question,
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Option 1: LLM
model = ChatOllama(model="llama3")
# Option 2: Multi-modal LLM
# model = LLaVA

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
# chain setup -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# chain setup adiexperiment -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Prompt template
# template1 = """
# Answer the question based only on the following context, and provide the source name and page number only once. If you fail to grasp the context ask a follow up question, based on what should be clarified, be extremely specific on what should be clarified. If the answer in the context has multiple categories ask a follow up question to get clarity on which category, by focusing on the context of the question. If the context is not clear provide the {question} to the next chain with token 69934.
# {context}
# Question: {question}
# """
# template2 = """
# Based on the context_2 and the context together, answer the question.
# {context}
# {context_2}
# Question: {question}
# """
# prompt1 = ChatPromptTemplate.from_template(template1)
# prompt2 = ChatPromptTemplate.from_template(template2)

# # Option 1: LLM
# model = ChatOllama(model="llama3")
# model_2 = ChatOllama(model="llava:latest")
# # Option 2: Multi-modal LLM
# # model = LLaVA

# # RAG pipeline
# chain1 = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt1
#     | model
#     | StrOutputParser()
# )

# chain2 = (
#     {"context": retriever, "context_2":chain1, "question":RunnablePassthrough()}
#     | prompt2
#     | model_2
#     | StrOutputParser()
# )

# chatbot -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.title('Chatbot')

if "messages" not in st.session_state:
	st.session_state["messages"] = [{"role": "assistant","content":"Hello! How can I help you today?"}]

# write messages
for msg in st.session_state.messages:
	st.chat_message(msg["role"]).write(msg["content"])

def generate_response():
	response = chain.stream(st.session_state.messages[-1]["content"])
	print(response)
	for partial_resp in response:
		token = partial_resp
		print(token)
		st.session_state["full_msg"] += token
		yield token

# def generate_response():
#     response = chain1.stream(st.session_state.messages[-1]["content"])
#     next_q=""
#     for txt in response:
#         next_q+=txt
#     print(next_q)
#     if ("?" or "clarification") in next_q:
#         extra_context = st.chat_input(next_q)
#         response = chain2.stream(next_q+" "+extra_context)
#         print(extra_context)
#     print(response)
#     for partial_resp in response:
#         token = partial_resp
#         print(token)
#         st.session_state["full_msg"] += token
#         yield token

if prompt := st.chat_input():
	st.session_state.messages.append({"role": "user", "content": prompt})
	st.chat_message("user").write(prompt)
	st.session_state["full_msg"] = ""
	st.chat_message("assistant").write_stream(generate_response())
	st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_msg"]})

# chatbot -----------------------------------------------------------------------------------------------------------------------------------------------------------------------