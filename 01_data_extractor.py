import pandas as pd
import json
import os
import re
import time
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import sys
import io

# 시스템 입출력을 UTF-8로 강제 고정 (모든 파일 공통 적용 권장)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
AUTH = (USER, PWD)

# --- [2. 데이터 구조 정의] ---
class Node(BaseModel):
    id: str = Field(description="고유 ID")
    label: str = Field(description="노드 타입 (Company, Report, Rating, Pillar, Theme, Content)")
    name: str = Field(description="엔티티의 실제 이름")

class Relationship(BaseModel):
    type: Literal["HAS_REPORT", "HAS_RATING", "HAS_CONTENT", "HAS_CATEGORY", "HAS_THEME"]
    start_node_id: str
    end_node_id: str

class GraphResponse(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

# --- [3. 유틸리티 함수] ---
def normalize_id(label, name):
    if pd.isna(name): return f"{label}_unknown"
    clean_name = re.sub(r'[^\w\s]', '', str(name)).strip().lower().replace(' ', '_')
    return f"{label}_{clean_name}"

def merge_graphs(raw_data_list):
    combined_nodes = {}
    combined_relationships = []
    
    for chunk in raw_data_list:
        if not chunk: continue
        id_map = {}
        for node in chunk.get('nodes', []):
            old_id = node['id']
            new_id = normalize_id(node['label'], node['name'])
            id_map[old_id] = new_id
            combined_nodes[new_id] = {"id": new_id, "label": node['label'], "name": node['name']}

        for rel in chunk.get('relationships', []):
            start_id = id_map.get(rel['start_node_id'], rel['start_node_id'])
            end_id = id_map.get(rel['end_node_id'], rel['end_node_id'])
            combined_relationships.append({"type": rel['type'], "start_node_id": start_id, "end_node_id": end_id})

    unique_rels = []
    seen_rels = set()
    for r in combined_relationships:
        rel_tuple = (r['type'], r['start_node_id'], r['end_node_id'])
        if rel_tuple not in seen_rels:
            seen_rels.add(rel_tuple)
            unique_rels.append(r)
            
    return {"nodes": list(combined_nodes.values()), "relationships": unique_rels}

# --- [4. 메인 실행 로직] ---
if __name__ == "__main__":
    FILE_PATH = os.getenv("DATA_FILE_PATH", "data/esg_database.csv") 
    OUTPUT_DIR = "output"
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint_graphs.json")
    FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_merged_graph_full.json")
    
    BATCH_SIZE = 5

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # api_key=OPENAI_API_KEY 로 수정 완료
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(GraphResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an ESG Knowledge Graph engineer. 
        Extract ALL entities and relationships from the provided CSV data.
        Multiple rows are provided. Process them all into one connected graph.
        Schema:
        - (Company)-[:HAS_REPORT]->(Report)
        - (Report)-[:HAS_RATING]->(Rating)
        - (Report)-[:HAS_CONTENT]->(Content)
        - (Report)-[:HAS_CATEGORY]->(Pillar)-[:HAS_THEME]->(Theme)"""),
        ("human", "Input CSV Data (Multiple Rows):\n{input_text}")
    ])

    # 데이터 로드
    try:
        df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(FILE_PATH, encoding="cp949")
        except Exception as e:
            print(f"❌ 파일을 찾을 수 없거나 인코딩 오류: {e}")
            df = pd.DataFrame()

    if not df.empty:
        all_raw_graphs = []
        processed_count = 0

        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                all_raw_graphs = json.load(f)
            processed_count = len(all_raw_graphs) * BATCH_SIZE 
            print(f"🔄 이전 기록 발견. 약 {processed_count}행 이후부터 시작합니다.")

        total_rows = len(df)
        chain = prompt | structured_llm

        for i in range(processed_count, total_rows, BATCH_SIZE):
            batch_df = df.iloc[i : i + BATCH_SIZE]
            print(f"[{min(i+BATCH_SIZE, total_rows)}/{total_rows}] 데이터 배치 처리 중...")

            batch_text = ""
            for idx, row in batch_df.iterrows():
                row_text = " / ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                batch_text += f"[Row {idx+1}]\n{row_text}\n\n"

            try:
                response = chain.invoke({"input_text": batch_text})
                all_raw_graphs.append(response.dict())
                
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_raw_graphs, f, ensure_ascii=False)
                    
            except Exception as e:
                print(f"❌ 배치 {i} 처리 중 오류: {e}")
                time.sleep(5)

        print("\n🧹 병합 및 최종 저장 중...")
        final_graph = merge_graphs(all_raw_graphs)

        with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_graph, f, ensure_ascii=False, indent=2)

        print(f"🎉 완료! 노드: {len(final_graph['nodes'])}, 관계: {len(final_graph['relationships'])}")