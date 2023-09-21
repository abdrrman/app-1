
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

st.title('My Psychologist v3')
#Get message from the user for the chat-based system
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  
        st.markdown(message["content"])

if message := st.chat_input("Enter your message"):
    with st.chat_message("user"):
        st.markdown(message)
    st.session_state.messages.append({"role": "user", "content": message})
#Generate a response from the psychologist based on the user's message
def psychologist_response(message):
    prompt = PromptTemplate(
        input_variables=['chat_history', 'message'], template='''You are a psychologist. Generate a response based on the user's message.

{chat_history}
User: {message}
Psychologist:'''
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="message")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key, temperature=0.7)
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    return chat_llm_chain
    

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    psychologist_response = ""
elif message:
    if 'chat_llm_chain' not in st.session_state:
        st.session_state.chat_llm_chain = psychologist_response(message)
    psychologist_response = st.session_state.chat_llm_chain.run(message=message)
else:
    psychologist_response = ""
#Display the psychologist's response to the user in a chat interface
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    # Simulate stream of response with milliseconds delay
    for chunk in psychologist_response.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
