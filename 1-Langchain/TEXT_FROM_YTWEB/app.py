import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

#streamlit app
st.set_page_config(page_title="please enter the page url you wamt to summarise",page_icon=":)")
st.title("summarise text from this page")
st.subheader("The URL is")

#get the Groq API and URL to be summarised
with st.sidebar:
    groq_api_key=st.text_input("GROQ API value", value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

#model setup and prompt template
llm=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

prompt_template=""" summarise the following content into 1000 words:
Content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])


#giving input through validations
if st.button("summarise the content from yt or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("please provide a valid URL. an yt or website")
    else:
        try:
            with st.spinner("waiting....."):
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                             headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                data=loader.load()

                #chain for summarisation
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(data)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")





