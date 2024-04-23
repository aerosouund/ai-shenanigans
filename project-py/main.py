from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import params


client = MongoClient(params.mongodb_string)
dbName = "langchain_demo"
collectionName = "coll"
collection = client[dbName][collectionName]


embdeddings = OpenAIEmbeddings()
vector_store = MongoDBAtlasVectorSearch(collection, embdeddings)

def query_data(query):
    docs = vector_store.similarity_search(query, K=1)
    as_output = docs[0].page_content

    llm = Ollama()
    retriever = vector_store.as_restriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)

    return as_output, retriever_output

