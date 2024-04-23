from pymongo import MongoClient
from langchain.embeddings import OllamaEmbeddings
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

loader = DirectoryLoader('./files', glob="./*.*", show_progress=True)
data = loader.load()

embdeddings = OllamaEmbeddings()
vector_store = MongoDBAtlasVectorSearch.from_documents(data, embdeddings, collection=collection)

