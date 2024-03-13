from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()

pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="recipe-book"

#Load index from Pinecode
docsearch=Pinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

#Llama2-7b GGML Model-Uncomment below line to use this model
llm=LlamaCpp(model_path="model/recipe_ingrident.Q4_K_M.gguf",n_batch=512,temperature=0.9)
#Mistral-7b GGML Model-Uncomment below line to use this model
#llm=LlamaCpp(model_path="model/recipe-ingredient-mistral-7b.Q4_K_M.gguf",n_batch=512,temperature=0.9)

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
