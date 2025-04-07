import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain,LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

#streamlit app
st.set_page_config(page_title="text to math problem solver",page_icon=":)")
st.title("this is my math problem solver")
st.subheader("the question is")

groq_api_key=st.text_input("GROQ API value", value="",type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

model=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

#this tool is to search wikipedia for the information to solve any math problem
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Atool for searching intrnet on math problem"
    )

#math tool, It implements the related calculation
math_chain=LLMMathChain.from_llm(llm=model)
calculator=Tool(
    name="calculator",
    func=math_chain.run,
    description="it is a tool to run the math problems and give solutions"
    )

prompt_template=""" you need to solve mathematical problems and 
display it stepwise 
Question:{question}
Answer:
"""
chain=LLMChain(llm=model,prompt=prompt_template)

#combine all tools to chain

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool to answer reasonong questions"
)

#initialize the agent

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi I am your math problem solver how can i help you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#function to generate response
"""def generate_response(question):
    response=assistant_agent.invoke({'input':question})
    return response"""

#start interaction

question=st.text_area("enter your question")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                         )
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")



