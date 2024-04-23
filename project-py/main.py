from pymongo import MongoClient
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import params
import gradio as gr
from gradio.themes.base import Base
import params


client = MongoClient(params.mongodb_string)
dbName = "langchain_demo"
collectionName = "coll"
collection = client[dbName][collectionName]


embdeddings = OllamaEmbeddings()
vector_store = MongoDBAtlasVectorSearch(collection, embdeddings)

def query_data(query):
    docs = vector_store.similarity_search(query, K=1)
    as_output = docs[0].page_content

    llm = Ollama(model="llama3")
    retriever = vector_store.as_restriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)

    return retriever_output

with gr.Blocks(theme=Base(), title="Ammar demo") as demo:
    gr.Markdown("Ammar demo")
    textbox=gr.Textbox(label="enter your question:")
    with gr.Row():
        button = gr.Button("submit", variant="primary")
    with gr.Column():
        output = gr.Textbox(lines=1, max_lines=10, label="model output")

    button.click(query_data, textbox, outputs=[output])


demo.launch()
