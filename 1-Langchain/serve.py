#pip install fastapi
#pip install uvicorn
#pip install langserve
#pip install sse_starlette

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langserve import add_routes
load_dotenv()

#intialise api key
groq_api_key=os.getenv("GROQ_API_KEY")

#build the model
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

#parser
parser=StrOutputParser()

#prompt
generic_template="Translate the message into {language}"
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",generic_template),("user","{text}")
    ]
)

#chain
chain=prompt|model|parser

#app definition
app=FastAPI(title="this is my LANG CHAIN SERVER",
            version="1.0",
            description="THIS IS MY FIRST APP USING LANGSERVE")

#adding chian routes
add_routes(
    app,
    chain,
    path="/chain"

)

#main function
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)

#python serve.py