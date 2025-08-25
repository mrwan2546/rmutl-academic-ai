import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from google.genai import types
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

loader_multiple_pages = WebBaseLoader(
    [
        "https://fileregis.rmutl.ac.th/academic/docs/faq-ent.html",
        "https://fileregis.rmutl.ac.th/academic/docs/faq-tc.html",
        "https://fileregis.rmutl.ac.th/academic/docs/faq-honor.html",
        "https://fileregis.rmutl.ac.th/academic/docs/faq-leave.html",
        "https://fileregis.rmutl.ac.th/academic/docs/faq-i.html"
    ]
)
docs = loader_multiple_pages.load()

text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size = 2048,
    length_function = len,
)
split_docs = text_splitter.split_documents(docs)

st.title("Academic RMUTL ChatBot Q&A")

# Async function for response
async def get_response(user_query, conversation_history):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        stream=True,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0) # Disable thinking
        ),
    )
    # embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory="./db"
    )
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "คุณคือเจ้าหน้าที่ระบบของ มหาวิทยาลัยเทคโนโลยีราชมงคลล้านนา โดยหน้าที่ของคุณคือตอบคำถามที่มาจากแหล่งข้อมูลที่หมอบให้เท่านั้น และคุณเป็นผู้หญิงที่สุภาพ โดยสามารถปรุงแต่งได้นิดหน่อย ห้ามลุด และหากเกิดข้อสงสัยหรือเว็บไซต์ทะเบียนกลาง ทุกครั้งควรแนบลิ้งก์ https://academic.rmutl.ac.th ไปด้วยเสมอ"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", user_query),
            ("ai", "จากข้อมูลที่ได้รับ: \n{context}\n\nคำตอบ:"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Stream only the plain text from "answer"
    async for chunk in retrieval_chain.astream(
        {"input": user_query, "chat_history": conversation_history}
    ):
        if "answer" in chunk:
            yield chunk["answer"]


# Wrapper to run async in Streamlit
def generate_response_with_stream(user_query, conversation_history):
    async def runner():
        async for text in get_response(user_query, conversation_history[-6:]):
            yield text
    return runner()


# Initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="สวัสดี ฉันเป็น AI Assistant มหาวิทยาลัยเทคโนโลยีราชมงคลล้านนา มีอะไรให้ฉันช่วยไหมคะ")
    ]

# Display old messages
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

# Chat input
if prompt := st.chat_input():
    # Add user message
    if not isinstance(st.session_state.messages[-1], HumanMessage):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)

    # Add assistant response
    if not isinstance(st.session_state.messages[-1], AIMessage):
        with st.chat_message("assistant"):
            response_text = st.write_stream(
                generate_response_with_stream(prompt, st.session_state.messages)
            )
        st.session_state.messages.append(AIMessage(content=response_text))
