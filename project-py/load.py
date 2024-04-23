from pymongo import MongoClient
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
import params


client = MongoClient(params.mongodb_string)
dbName = "langchain_demo"
collectionName = "coll"
collection = client[dbName][collectionName]

loader = DirectoryLoader('./files', glob="./*.txt", show_progress=True)
data = loader.load()

embdeddings = OllamaEmbeddings()
vector_store = MongoDBAtlasVectorSearch.from_documents(data, embdeddings, collection=collection)

