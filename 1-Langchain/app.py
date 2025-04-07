import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

load_dotenv()


os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

#prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are student. Provide me answers based on questions"),
        ("user","Question:{question}")
    ]
)

#streamlit framework

st.title("Langchain demo with OLLAMA")
input_text=st.text_input("What is your question?")

##ollama Llama2 model
llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

#now if i put any text in the input text the below will get executed and run the llm model
if input_text:
    st.write(chain.invoke({"question":input_text}))

#streamlit run app.py
#make sure you are in correct folder location before running the model