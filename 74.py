import os
import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document
import time
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)

st.title('MathApp')
#Get file path from the user for the document file
with st.form(key='file_upload'):
    uploaded_file = st.file_uploader("Upload Document File", type=["docx", "pdf"])
    submit_button = st.form_submit_button(label='Submit File')
    if submit_button:
        if uploaded_file is not None:
            # Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                file_path = temp_file.name # it shows the file path
                st.session_state['file_path'] = file_path
        else:
            file_path = ''

#Load the document file as Document from the file path
def load_document(file_path):
    if file_path.endswith(".txt"):
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path)
    elif file_path.endswith(".pdf"):
        from langchain.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="fast")
    elif file_path.endswith(".pptx"):
        from langchain.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(file_path, mode="elements", strategy="fast")
    elif file_path.endswith(".csv"):
        from langchain.document_loaders.csv_loader import UnstructuredCSVLoader
        loader = UnstructuredCSVLoader(file_path, mode="elements")
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        from langchain.document_loaders.excel import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    else:
        raise ValueError("Unsupported file format")

    docs = loader.load()
    return docs

if file_path:
    document = load_document(file_path)
else:
    document = ''
#Convert Document to string content
document_str = "".join([doc.page_content for doc in document])
#Extract the math problem from the document using OCR
def mathProblemExtractor(document_str):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are an OCR assistant. Your task is to extract the math problem from the given document."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please extract the math problem from the following document: '{document_str}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(document_str=document_str)
    return result # returns string   


if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    math_problem = ""
elif document_str:
    math_problem = mathProblemExtractor(document_str)
else:
    math_problem = ""
#Solve the math problem step by step
def mathSolver(math_problem):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are a virtual math teacher, capable of solving any given math problem."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """The problem is: {math_problem}. Please solve it and show the steps."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(math_problem=math_problem)
    return result # returns string   


if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    solution = ""
elif math_problem:
    solution = mathSolver(math_problem)
else:
    solution = ""
#Display the solution to the user
if solution:
    st.code(solution)