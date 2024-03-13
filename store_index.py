from langchain.vectorstores import Pinecone
import pinecone
from src.helper import load_pdf, load_web, text_split, download_hugging_face_embeddings
from src.helper import load_pdf
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

pdf_data = load_pdf("datafile")
web_list = ["https://www.vegrecipesofindia.com/", "https://www.indianhealthyrecipes.com/", "https://www.allrecipes.com/recipes/", "https://www.bbcgoodfood.com/recipes", "https://www.thecookierookie.com/"]
web_data = load_web(web_list)
extracted_data = pdf_data+web_data


text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="recipe-book"

docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

