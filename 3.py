
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

st.title('Admin')
#Get resume files from the user
uploaded_file = st.file_uploader("Upload Resume File", type=["txt"])
if uploaded_file is not None:
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        resume_files = temp_file.name # it shows the file path
else:
    resume_files = ''
#Load the resume files as Document from the file path
from langchain.document_loaders import UnstructuredPDFLoader

def load_resumes(resume_files):
    resume_docs = []
    for path in resume_files:
        loader = UnstructuredPDFLoader(path, mode="elements", strategy="fast")
        docs = loader.load()
        resume_docs.append(docs)
    return resume_docs

if resume_files:
    resume_docs = load_resumes(resume_files)
else:
    resume_docs = ''
#Convert the Document objects of resumes to string
resume_strings = "".join([doc.page_content for doc in resume_docs])
#Analyze the resumes
def resumeAnalyzer(resume_strings):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are an assistant designed to analyze resumes. Your task is to extract relevant information from the given resumes."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please analyze the following resumes: '{resume_strings}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(resume_strings=resume_strings)
    return result # returns string   


if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    analyzed_resumes = ""
elif resume_strings:
    analyzed_resumes = resumeAnalyzer(resume_strings)
else:
    analyzed_resumes = ""
#Show the analyzed resumes to the user
if analyzed_resumes:
    st.table(analyzed_resumes)
#Get job posting text from the user
job_posting_text = st.text_area("Enter job posting text")
#Optimize the job posting
def jobPostingOptimizer(job_posting_text):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0.7
    )
    system_template = """You are an assistant designed to optimize job postings. Your task is to make the job posting more appealing and effective."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Here is the job posting text: '{job_posting_text}'. Please optimize it."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(job_posting_text=job_posting_text)
    return result # returns string   


if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    optimized_job_posting = ""
elif job_posting_text:
    optimized_job_posting = jobPostingOptimizer(job_posting_text)
else:
    optimized_job_posting = ""
#Show the optimized job posting to the user
if optimized_job_posting:
    st.success(optimized_job_posting)
#Get candidate queries from the user
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  
        st.markdown(message["content"])

if candidate_queries := st.chat_input("Enter the candidate queries"):
    with st.chat_message("user"):
        st.markdown(candidate_queries)
    st.session_state.messages.append({"role": "user", "content": candidate_queries})
#Respond to the candidate queries
def respondToCandidateQueries(candidate_queries):
    prompt = PromptTemplate(
        input_variables=['chat_history', 'candidate_queries'], template='''You are a chatbot having a conversation with a job candidate. Respond to the candidate's queries as accurately and professionally as possible.

{chat_history}
Candidate: {candidate_queries}
Chatbot:'''
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="candidate_queries")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key, temperature=0)
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    return chat_llm_chain
    

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    candidate_responses = ""
elif candidate_queries:
    if 'chat_llm_chain' not in st.session_state:
        st.session_state.chat_llm_chain = respondToCandidateQueries(candidate_queries)
    candidate_responses = st.session_state.chat_llm_chain.run(candidate_queries=candidate_queries)
else:
    candidate_responses = ""
#Show the responses to the user
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    # Simulate stream of response with milliseconds delay
    for chunk in candidate_responses.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
#Get employee feedback from the user
employee_feedback = st.text_area("Enter your feedback")
#Analyze the employee feedback
def feedbackAnalyzer(employee_feedback):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are an AI assistant designed to analyze employee feedback."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please analyze the following employee feedback: '{employee_feedback}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(employee_feedback=employee_feedback)
    return result # returns string   


if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    analyzed_feedback = ""
elif employee_feedback:
    analyzed_feedback = feedbackAnalyzer(employee_feedback)
else:
    analyzed_feedback = ""
#Show the analyzed feedback to the user
if analyzed_feedback:
    st.write(analyzed_feedback)
#Get employee queries from the user
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  
        st.markdown(message["content"])

if employee_queries := st.chat_input("Enter your queries"):
    with st.chat_message("user"):
        st.markdown(employee_queries)
    st.session_state.messages.append({"role": "user", "content": employee_queries})
#Respond to the employee queries
def respondEmployeeQueries(employee_queries):
    prompt = PromptTemplate(
        input_variables=['chat_history', 'employee_queries'], template='''You are a chatbot designed to respond to employee queries. Your task is to provide accurate and helpful responses to the questions asked by the employees.

{chat_history}
Employee: {employee_queries}
Chatbot:'''
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="employee_queries")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key, temperature=0)
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    return chat_llm_chain
    

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    employee_responses = ""
elif employee_queries:
    if 'chat_llm_chain' not in st.session_state:
        st.session_state.chat_llm_chain = respondEmployeeQueries(employee_queries)
    employee_responses = st.session_state.chat_llm_chain.run(employee_queries=employee_queries)
else:
    employee_responses = ""
#Show the responses to the user
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    # Simulate stream of response with milliseconds delay
    for chunk in employee_responses.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
#Get policy documents from the user
# Get policy documents from the user
uploaded_file = st.file_uploader("Upload Policy Document", type=["txt"])
if uploaded_file is not None:
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        policy_documents = temp_file.name # it shows the file path
else:
    policy_documents = ''
#Load the policy documents as Document from the file path
from langchain.document_loaders import TextLoader, WebBaseLoader, OnlinePDFLoader, UnstructuredPDFLoader, UnstructuredPowerPointLoader, UnstructuredCSVLoader, UnstructuredExcelLoader

def load_policy_documents(policy_documents):
    if policy_documents.endswith('.txt'):
        loader = TextLoader(policy_documents)
    elif policy_documents.startswith('http'):
        if policy_documents.endswith('.pdf'):
            loader = OnlinePDFLoader(policy_documents)
        else:
            loader = WebBaseLoader(policy_documents)
    elif policy_documents.endswith('.pdf'):
        loader = UnstructuredPDFLoader(policy_documents, mode="elements", strategy="fast")
    elif policy_documents.endswith('.ppt') or policy_documents.endswith('.pptx'):
        loader = UnstructuredPowerPointLoader(policy_documents, mode="elements", strategy="fast")
    elif policy_documents.endswith('.csv'):
        loader = UnstructuredCSVLoader(policy_documents, mode="elements")
    elif policy_documents.endswith('.xls') or policy_documents.endswith('.xlsx'):
        loader = UnstructuredExcelLoader(policy_documents, mode="elements")
    else:
        raise ValueError("Unsupported file type")
    
    docs = loader.load()
    return docs

if policy_documents:
    policy_docs = load_policy_documents(policy_documents)
else:
    policy_docs = ''
#Convert the Document objects of policies to string
policy_strings = "".join([doc.page_content for doc in policy_docs])
#Analyze the policy documents
def policyAnalyzer(policy_strings):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are an AI assistant designed to analyze policy documents."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please analyze the following policy document: '{policy_strings}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(policy_strings=policy_strings)
    return result # returns string   


if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    analyzed_policies = ""
elif policy_strings:
    analyzed_policies = policyAnalyzer(policy_strings)
else:
    analyzed_policies = ""
#Show the analyzed policies to the user
if analyzed_policies:
    st.table(analyzed_policies)
#Get employee communications from the user
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  
        st.markdown(message["content"])

if employee_communications := st.chat_input("Enter the employee communications"):
    with st.chat_message("user"):
        st.markdown(employee_communications)
    st.session_state.messages.append({"role": "user", "content": employee_communications})
#Monitor the employee communications for policy violations
def monitorPolicyViolations(employee_communications):
    prompt = PromptTemplate(
        input_variables=['chat_history', 'employee_communications'], template='''You are an AI system monitoring employee communications for policy violations. Analyze the following conversation and identify any potential policy violations.

{employee_communications}

AI Monitor:'''
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="employee_communications")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key, temperature=0)
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    return chat_llm_chain
    

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    policy_violations = ""
elif employee_communications:
    if 'chat_llm_chain' not in st.session_state:
        st.session_state.chat_llm_chain = monitorPolicyViolations(employee_communications)
    policy_violations = st.session_state.chat_llm_chain.run(employee_communications=employee_communications)
else:
    policy_violations = ""
#Show the policy violations to the user
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    # Simulate stream of response with milliseconds delay
    for chunk in policy_violations.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
