import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS # FAISS로 변경
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 환경 변수 로드
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "2024_KB_부동산_보고서_최종.pdf")
# PDF 처리 함수 (@st.cache_resource로 한 번만 로드되게 캐싱)

@st.cache_resource
def process_pdf():
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 벡터 스토어 초기화
@st.cache_resource
def initialize_vectorstore():
    # chunks = process_pdf()
    embeddings = OpenAIEmbeddings()
    if not os.path.exists("./faiss_db"):
        chunks = process_pdf()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("./faiss_db")  # FAISS DB를 로컬에 저장
    # 로컬 경로에 FAISS DB 저장
    # 기존에는 매번 Chroma DB를 새로 생성했지만, 이제는 FAISS.load_local을 사용해 이미 만들어진 로컬 FAISS DB를 불러오도록 변경
    return FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)

# 체인 초기화
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

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            chat_history=lambda x: x.get("chat_history", [])[-4:]  # 대화 기록을 프롬프트에 포함
            # base_chain 내부에 chat_history=lambda x: x.get("chat_history", [])[-4:] 코드가 추가되어 대화 이력을 최근 4개까지만 유지하도록 수정
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

# Streamlit UI 메인 함수
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 강남 멋쟁이들의 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 세션 상태(대화 기록) 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 채팅 기록 화면에 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        # 사용자 메시지 화면 표시 및 저장
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 체인 불러오기
        chain = initialize_chain()

        # AI 응답 생성 및 화면 표시
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
            st.markdown(response)

        # AI 응답을 대화 기록에 저장
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    from pyngrok import ngrok
    
    # Streamlit 기본 포트(8501)로 ngrok 터널링 연결
    public_url = ngrok.connect(8501)
    print("앱 접속 URL:", public_url)
    
    main()