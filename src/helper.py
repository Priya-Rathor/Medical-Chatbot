from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai



genai.configure(api_key="AIzaSyDUiT3yPTTo2nmoPRj-hpo2r2OyrPH5cqs")
#-------------------------------------------------------------------------
#                Extract data from the pdf
#-------------------------------------------------------------------------


def load_pdf(data):
    loader = DirectoryLoader(data,glob ='*.pdf',loader_cls = PyPDFLoader)

    documents = loader.load()

    return documents


extracted_data = load_pdf("C:/Users/USER/Projects/MedicalChatbot/data/")

print(extracted_data)


