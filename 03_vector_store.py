import os
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
from neo4j_graphrag.indexes import create_vector_index
from dotenv import load_dotenv

import sys
import io

# 시스템 입출력을 UTF-8로 강제 고정 (모든 파일 공통 적용 권장)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


load_dotenv()

# 환경 변수에서 값 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI 임베딩 모델 설정
embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_key=OPENAI_API_KEY
)

def setup_vector_index():
    # 드라이버 설정 (환경 변수 사용)
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    # --- 1단계: 인덱스 생성 ---
    # Company와 Content 두 곳 모두 생성 (검색 성능 향상)
    indices = [
        {"name": "company_name_index", "label": "Company", "prop": "name"},
        {"name": "content_text_index", "label": "Content", "prop": "name"}
    ]

    for idx in indices:
        print(f"✨ {idx['name']} 인덱스 생성 중...")
        try:
            create_vector_index(
                driver,
                index_name=idx['name'],
                label=idx['label'],
                embedding_property="embedding",
                dimensions=1536, # OpenAI text-embedding-3-small 모델의 차원
                similarity_fn="cosine",
            )
            print(f"✅ {idx['name']} 생성 완료!")
        except Exception:
            print(f"알림: {idx['name']}이 이미 존재하거나 생성을 건너뜁니다.")

    # --- 2단계: 임베딩 업데이트 (배치 처리) ---
    target_labels = ["Company", "Content"]
    
    for label in target_labels:
        print(f"🧠 {label} 노드 임베딩 업데이트 중...")
        with driver.session() as session:
            # 임베딩이 아직 없는 노드만 추출
            result = session.run(
                f"MATCH (n:{label}) WHERE n.embedding IS NULL RETURN n.id as id, n.name as name"
            )
            
            for record in result:
                node_id = record["id"]
                text_to_embed = record["name"]
                
                if not text_to_embed:
                    continue
                
                # 텍스트를 숫자로 변환 (Embedding)
                try:
                    embedding_vector = embed_model.embed_query(text_to_embed)
                    
                    # 생성된 벡터를 DB의 'embedding' 속성에 저장
                    session.run(
                        f"MATCH (n:{label} {{id: $id}}) "
                        f"CALL db.create.setNodeVectorProperty(n, 'embedding', $vector)",
                        {"id": node_id, "vector": embedding_vector}
                    )
                    print(f"   - [{label}] '{text_to_embed[:10]}...' 임베딩 완료")
                except Exception as e:
                    print(f"❌ '{text_to_embed[:10]}' 임베딩 중 오류 발생: {e}")

    print("🎉 모든 임베딩 작업이 완료되었습니다!")
    driver.close()

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ 오류: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    else:
        setup_vector_index()