# -*- coding: utf-8 -*-
import json
import asyncio
import os
import sys
import re
from neo4j import GraphDatabase, TRUST_ALL_CERTIFICATES
from pydantic import validate_call
from dotenv import load_dotenv

import sys
import io

# 시스템 입출력을 UTF-8로 강제 고정 (모든 파일 공통 적용 권장)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Neo4j GraphRAG 관련 컴포넌트
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, KGWriterModel


load_dotenv()


URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD")
DB_NAME = os.getenv("NEO4J_DATABASE", "neo4j")

# [시스템 설정] 터미널 인코딩 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError: pass

# --- 2. Custom Writer 클래스 ---
class Neo4jCreateWriter(KGWriter):
    def __init__(self, driver, neo4j_database="neo4j"):
        self.driver = driver
        self.neo4j_database = neo4j_database

    def _prepare_db(self):
        """DB 제약 조건 생성 및 인덱싱 최적화"""
        with self.driver.session(database=self.neo4j_database) as session:
            # ID 중복 방지를 위한 제약 조건 설정
            labels = ["Company", "Report", "Rating", "Pillar", "Theme", "Content"]
            for label in labels:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    @validate_call
    async def run(self, graph: Neo4jGraph) -> KGWriterModel:
        try:
            self._prepare_db()
            
            with self.driver.session(database=self.neo4j_database) as session:
                # 1. 노드 적재 (MERGE 사용)
                print(f"📦 {len(graph.nodes)}개 노드 적재 시작...")
                for node in graph.nodes:
                    cypher = f"MERGE (n:`{node.label}` {{id: $id}}) SET n += $props"
                    session.run(cypher, {"id": node.id, "props": node.properties})

                # 2. 관계 적재
                print(f"🔗 {len(graph.relationships)}개 관계 연결 시작...")
                for rel in graph.relationships:
                    cypher = f"""
                    MATCH (a {{id: $start_id}}), (b {{id: $end_id}})
                    MERGE (a)-[r:`{rel.type}`]->(b)
                    SET r += $props
                    """
                    session.run(cypher, {
                        "start_id": rel.start_node_id,
                        "end_id": rel.end_node_id,
                        "props": rel.properties or {}
                    })

            return KGWriterModel(status="SUCCESS", metadata={"nodes": len(graph.nodes), "rels": len(graph.relationships)})
        except Exception as e:
            return KGWriterModel(status="FAILURE", metadata={"error": str(e)})

# --- 3. 실행 메인 함수 ---
async def main():
    # [경로 설정] output 폴더 내부의 JSON 파일 자동 탐색
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # .env에서 경로를 가져오거나 기본값 사용
    input_path = os.path.join(base_dir, "output", "final_merged_graph_full.json")
    
    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return

    # 1. JSON 로드
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Graph 객체 변환
    nodes = []
    for n in data.get("nodes", []):
        props = {k: v for k, v in n.items() if k not in ["id", "label"]}
        nodes.append(Neo4jNode(id=n["id"], label=n["label"], properties=props))
    
    relationships = []
    for rel in data.get("relationships", []):
        props = {k: v for k, v in rel.items() if k not in ["start_node_id", "end_node_id", "type"]}
        relationships.append(
            Neo4jRelationship(
                start_node_id=rel["start_node_id"], 
                end_node_id=rel["end_node_id"], 
                type=rel["type"],
                properties=props
            )
        )
    
    graph_obj = Neo4jGraph(nodes=nodes, relationships=relationships)

    # 3. 드라이버 설정 (환경 변수 사용)
    driver = GraphDatabase.driver(
        URI, 
        auth=(USER, PWD), 
        encrypted=False, 
        trust=TRUST_ALL_CERTIFICATES
    )
    
    writer = Neo4jCreateWriter(driver, neo4j_database=DB_NAME)
    
    print(f"🚀 Neo4j({URI}) 적재 프로세스를 시작합니다...")
    result = await writer.run(graph_obj)
    
    print("-" * 40)
    if result.status == "SUCCESS":
        print(f"✨ 적재 성공!")
        print(f"📊 통계: 노드 {result.metadata['nodes']}개 / 관계 {result.metadata['rels']}개")
    else:
        print(f"❌ 적재 실패: {result.metadata.get('error')}")
    print("-" * 40)
    
    driver.close()

if __name__ == "__main__":
    asyncio.run(main())