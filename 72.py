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

st.title('stock_hist')

def stockDataRetriever(stock_id):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are a stock data analyst. Your task is to retrieve historical stock data based on the stock ID: '{stock_id}'."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please retrieve the historical stock data for the stock with ID: {stock_id}."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(stock_id=stock_id)
    return result # returns string   

def chartGenerator(stock_data):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are a data analyst tasked with generating a visualized chart from the given stock data {stock_data}."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please create a chart based on the provided stock data {stock_data}."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(stock_data=stock_data)
    return result # returns string   

with st.form(key='stock_hist'):
    stock_id = st.text_input("Enter stock ID")
    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
            stock_data = ""
        elif stock_id:
            stock_data = stockDataRetriever(stock_id)
        else:
            stock_data = ""
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
            chart = ""
        elif stock_data:
            chart = chartGenerator(stock_data)
        else:
            chart = ""
        if chart:
            st.pyplot(chart)