import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "2024_KB_부동산_보고서_최종.pdf")

api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def process_pdf():
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=api_key)

    faiss_path = os.path.join(BASE_DIR, "faiss_db")

    if not os.path.exists(faiss_path):
        chunks = process_pdf()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(faiss_path)

    return FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

컨텍스트: {context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            chat_history=lambda x: x.get("chat_history", [])[-4:]
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

def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 강남 멋쟁이들의 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        chain = initialize_chain()

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