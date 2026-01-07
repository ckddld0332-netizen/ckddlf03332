# -*- coding: utf-8 -*-
import sys
import io
import streamlit as st
import os
import openai
from neo4j import GraphDatabase, TRUST_ALL_CERTIFICATES
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from dotenv import load_dotenv

# [1. 인코딩 및 환경 변수 로드]
# 윈도우 환경에서 한글 깨짐 및 ASCII 에러 방지
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# .env 파일 로드
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD")
AUTH = (USER, PWD)

# --- [2. 페이지 설정 및 초기화] ---
st.set_page_config(page_title="ESG GraphRAG Explorer", page_icon="🌿", layout="wide")

# ★ 중요: session_state 초기화 (오류 방지를 위해 가장 상단에 배치)
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_rag_engine():
    """RAG 엔진 초기화: 검색 성공률을 높이기 위한 설정"""
    driver = GraphDatabase.driver(URI, auth=AUTH, encrypted=False, trust=TRUST_ALL_CERTIFICATES)
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)

    # 예시 데이터: Theme과 Pillar의 관계를 명시
    esg_examples = [
        "Question: 'Climate Change' 테마는 어떤 Pillar에 속해 있어? Answer: MATCH (t:Theme) WHERE toLower(t.name) CONTAINS toLower('Climate Change') MATCH (p:Pillar)-[:HAS_THEME]->(t) RETURN p.name, t.name",
        "Question: NetApp의 환경 등급? Answer: MATCH (c:Company) WHERE toLower(c.name) CONTAINS toLower('NetApp') MATCH (c)-[:HAS_REPORT]->(rep)-[:HAS_RATING]->(rat) RETURN c.name, rat.name"
    ]
    
    # 그래프 구조(Schema)를 AI에게 더 명확히 설명
    custom_prompt = """
    Task: Generate a Cypher query to explore an ESG Knowledge Graph.
    
    Rules:
    1. A Theme belongs to a Pillar: (p:Pillar)-[:HAS_THEME]->(t:Theme)
    2. A Report has Categories: (rep:Report)-[:HAS_CATEGORY]->(p:Pillar)
    3. A Company has Reports: (c:Company)-[:HAS_REPORT]->(rep:Report)
    4. ALWAYS use 'toLower(node.name) CONTAINS toLower("search_term")' for flexible filtering.

    Graph Schema:
    - (p:Pillar)-[:HAS_THEME]->(t:Theme)
    - (rep:Report)-[:HAS_CATEGORY]->(p:Pillar)
    - (c:Company)-[:HAS_REPORT]->(rep:Report)

    {examples}
    Question: {query_text}
    """

    retriever = Text2CypherRetriever(driver=driver, llm=llm, examples=esg_examples, custom_prompt=custom_prompt)
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return openai_client, retriever

def generate_answer(client, retriever, user_question):
    """답변 생성 및 검색 결과 확인"""
    try:
        result = retriever.search(query_text=str(user_question))
        cypher_used = result.metadata.get("cypher", "")
        items = getattr(result, 'items', [])

        if not items:
            return None, cypher_used

        context = "\n".join([str(i) for i in items])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 ESG 전문가입니다. 제공된 데이터를 기반으로 한국어로 답변하세요."},
                {"role": "user", "content": f"질문: {user_question}\n\n데이터: {context}"}
            ]
        )
        return response.choices[0].message.content, cypher_used

    except Exception as e:
        return f"❌ 분석 중 오류 발생: {str(e)}", "Error"

# --- [3. 메인 UI 구성] ---
st.title("🌿 ESG GraphRAG Explorer")
st.markdown("Neo4j 그래프 데이터를 기반으로 ESG 정보를 분석합니다.")

# 엔진 초기화
try:
    if not OPENAI_API_KEY:
        st.error("🔑 .env 파일에 OPENAI_API_KEY를 설정해주세요.")
        st.stop()
    client, retriever = init_rag_engine()
except Exception as e:
    st.error(f"⚠️ 엔진 초기화 실패: {str(e)}")
    st.stop()

# 기존 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    # 1. 사용자 질문 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 어시스턴트 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("그래프 데이터를 분석 중입니다..."):
            answer, cypher_used = generate_answer(client, retriever, prompt)
            
            if answer is None:
                st.error("❌ 데이터를 찾지 못했습니다. 검색 조건을 바꿔보세요.")
            else:
                st.markdown(answer)
            
            # 실행된 쿼리 표시
            with st.expander("🛠️ 실행된 Cypher 쿼리 확인"):
                st.code(cypher_used, language="cypher")
            
            # 답변 저장
            st.session_state.messages.append({"role": "assistant", "content": answer if answer else "검색 결과 없음"})