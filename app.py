from flask import Flask, render_template, jsonify, request
from src.helper import download_genai_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINICONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_genai_embeddings()

#-------------------------------------------------------------------------
#                     Initializeing the Pinecone
#------------------------------------------------------------------------


pinecone.init(api_key = PINECONE_API_KEY,environment=PINECONE_API_ENV)


index_name = "medical-bot"


#--------------------------------------------------------------------------
#                   Loading the index  
#--------------------------------------------------------------------------


docsearch = Pinecone.from_existing_index(index_name,embeddings)

PROMPT = PromptTemplate(template=prompt_template,input_variables=["context","question"])

chain_type_kwargs = {"prompt":PROMPT}


llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})



qa=RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriver = docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documnets = True,
    chain_type_kwargs=chain_type_kwargs
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg 
    print(input)
    result = qa({"query":input})
    print("Response :",result["result"])

    return str(result["result"])