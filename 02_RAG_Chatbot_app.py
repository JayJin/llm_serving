# 02_RAG_Chatbot_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 변경
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  # 변경
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory  # 변경
from langchain_community.embeddings import HuggingFaceEmbeddings

# PDF 처리 함수
@st.cache_resource
def process_pdf():
    loader = PyPDFLoader("./data/국민주택채권 자주하는 질문(FAQ).pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 벡터 스토어 초기화
@st.cache_resource
def initialize_vectorstore():
    chunks = process_pdf()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return Chroma.from_documents(chunks, embeddings)

# 체인 초기화
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 국민주택채권 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

    컨텍스트: {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(model_name="qwen2.5-3b-instruct-q4_k_m.gguf", temperature=0.2, base_url="http://localhost:8002/v1", api_key="EMPTY")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="question",
        history_messages_key="chat_history",
    )

# Streamlit UI
def main():
    st.set_page_config(page_title="국민주택채권 챗봇", page_icon="🏠")
    st.title("🏠 국민주택채권 AI 어드바이저")
    st.caption("국민주택채권 자주하는 질문(FAQ) 기반 질의응답 시스템")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("국민주택채권 관련 질문을 입력하세요"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 체인 초기화
        chain = initialize_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()