"""
제조업 RAG 시스템
- RAG1: DB 스키마 정보
- RAG2: KPI 공식 정보  
- RAG3: Golden SQL 쿼리
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
import io
import sys

# 표준 출력(stdout)을 UTF-8로 인코딩하도록 재설정
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', line_buffering=True)
    except:
        pass

@dataclass
class Document:
    """RAG 문서 클래스"""
    doc_id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class SimpleEmbedding:
    """
    간단한 TF-IDF 기반 임베딩 (실제 환경에서는 sentence-transformers 등 사용)
    """
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.vocab_size = 0
        
    def _tokenize(self, text: str) -> List[str]:
        """한글/영문 토큰화"""
        # 소문자 변환 및 특수문자 처리
        text = text.lower()
        # 한글, 영문, 숫자만 추출
        tokens = re.findall(r'[가-힣]+|[a-z]+|[0-9]+', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """어휘 사전 및 IDF 계산"""
        # 어휘 사전 구축
        all_tokens = set()
        doc_freq = {}
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            all_tokens.update(tokens)
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # 어휘 인덱스 생성
        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self.vocab_size = len(self.vocab)
        
        # IDF 계산
        n_docs = len(documents)
        for token, freq in doc_freq.items():
            self.idf[token] = np.log((n_docs + 1) / (freq + 1)) + 1
            
    def transform(self, text: str) -> np.ndarray:
        """텍스트를 벡터로 변환 (TF-IDF)"""
        tokens = self._tokenize(text)
        vector = np.zeros(self.vocab_size)
        
        # TF 계산
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        # TF-IDF 벡터 생성
        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                vector[idx] = count * self.idf.get(token, 1)
        
        # L2 정규화
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector


class RAGSystem:
    """RAG 시스템 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.documents: List[Document] = []
        self.embedder = SimpleEmbedding()
        
    def add_document(self, doc_id: str, content: str, metadata: Dict):
        """문서 추가"""
        doc = Document(doc_id=doc_id, content=content, metadata=metadata)
        self.documents.append(doc)
        
    def build_index(self):
        """임베딩 인덱스 구축"""
        # 모든 문서 내용으로 어휘 학습
        contents = [doc.content for doc in self.documents]
        self.embedder.fit(contents)
        
        # 각 문서 임베딩
        for doc in self.documents:
            doc.embedding = self.embedder.transform(doc.content)
            
        print(f"[{self.name}] 인덱스 구축 완료: {len(self.documents)}개 문서, 어휘 크기: {self.embedder.vocab_size}")
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """유사도 기반 검색"""
        query_embedding = self.embedder.transform(query)
        
        # 코사인 유사도 계산
        results = []
        for doc in self.documents:
            if doc.embedding is not None:
                similarity = np.dot(query_embedding, doc.embedding)
                results.append((doc, similarity))
        
        # 유사도 순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class ManufacturingRAGSystem:
    """제조업 통합 RAG 시스템"""
    
    def __init__(self):
        self.rag1_schema = RAGSystem("RAG1-Schema")
        self.rag2_kpi = RAGSystem("RAG2-KPI")
        self.rag3_golden_sql = RAGSystem("RAG3-GoldenSQL")
        
    def load_schema_data(self, filepath: str):
        """스키마 데이터 로드 (RAG1)"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for table in data['tables']:
            # 테이블 정보를 검색 가능한 텍스트로 변환
            columns_text = ", ".join([
                f"{col['name']}({col['type']}): {col['description']}" 
                for col in table['columns']
            ])
            
            content = f"""
            테이블명: {table['table_name']}
            스키마: {table['schema']}
            설명: {table['description']}
            컬럼: {columns_text}
            관계: {', '.join(table.get('relationships', []))}
            """
            
            self.rag1_schema.add_document(
                doc_id=f"schema_{table['table_name']}",
                content=content,
                metadata={
                    "table_name": table['table_name'],
                    "schema": table['schema'],
                    "columns": table['columns'],
                    "type": "schema"
                }
            )
        
        self.rag1_schema.build_index()
        print(f"스키마 RAG 로드 완료: {len(data['tables'])}개 테이블")
        
    def load_kpi_data(self, filepath: str):
        """KPI 데이터 로드 (RAG2)"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for kpi in data['kpis']:
            # KPI 정보를 검색 가능한 텍스트로 변환
            sub_formulas = ""
            if 'sub_formulas' in kpi:
                sub_formulas = " / ".join([f"{k}: {v}" for k, v in kpi['sub_formulas'].items()])
            
            content = f"""
            KPI명: {kpi['kpi_name']} ({kpi['kpi_name_kr']})
            카테고리: {kpi['category']}
            설명: {kpi['description']}
            공식: {kpi['formula']}
            세부공식: {sub_formulas}
            단위: {kpi['unit']}
            목표값: {kpi['target_value']}
            관련테이블: {', '.join(kpi['related_tables'])}
            해석: {kpi.get('interpretation', '')}
            """
            
            self.rag2_kpi.add_document(
                doc_id=f"kpi_{kpi['kpi_id']}",
                content=content,
                metadata={
                    "kpi_id": kpi['kpi_id'],
                    "kpi_name": kpi['kpi_name'],
                    "kpi_name_kr": kpi['kpi_name_kr'],
                    "formula": kpi['formula'],
                    "sql_example": kpi.get('sql_example', ''),
                    "related_tables": kpi['related_tables'],
                    "type": "kpi"
                }
            )
        
        self.rag2_kpi.build_index()
        print(f"KPI RAG 로드 완료: {len(data['kpis'])}개 KPI")
        
    def load_golden_sql_data(self, filepath: str):
        """Golden SQL 데이터 로드 (RAG3)"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for query in data['queries']:
            content = f"""
            자연어: {query['natural_language']}
            키워드: {', '.join(query['keywords'])}
            카테고리: {query['category']}
            SQL: {query['sql']}
            설명: {query['explanation']}
            """
            
            self.rag3_golden_sql.add_document(
                doc_id=f"sql_{query['query_id']}",
                content=content,
                metadata={
                    "query_id": query['query_id'],
                    "natural_language": query['natural_language'],
                    "keywords": query['keywords'],
                    "sql": query['sql'],
                    "explanation": query['explanation'],
                    "type": "golden_sql"
                }
            )
        
        self.rag3_golden_sql.build_index()
        print(f"Golden SQL RAG 로드 완료: {len(data['queries'])}개 쿼리")
        
    def search_all(self, query: str, top_k: int = 3) -> Dict[str, List]:
        """모든 RAG에서 검색"""
        results = {
            "schema": self.rag1_schema.search(query, top_k),
            "kpi": self.rag2_kpi.search(query, top_k),
            "golden_sql": self.rag3_golden_sql.search(query, top_k)
        }
        return results
    
    def generate_context(self, query: str, top_k: int = 3) -> str:
        """쿼리 생성을 위한 컨텍스트 생성"""
        results = self.search_all(query, top_k)
        
        context_parts = []
        
        # 스키마 정보
        context_parts.append("=== 관련 테이블 스키마 ===")
        for doc, score in results["schema"]:
            if score > 0.1:  # 임계값 이상만
                context_parts.append(f"\n[테이블: {doc.metadata['table_name']}] (유사도: {score:.3f})")
                context_parts.append(f"스키마: {doc.metadata['schema']}")
                context_parts.append("컬럼:")
                for col in doc.metadata['columns']:
                    pk = " (PK)" if col.get('primary_key') else ""
                    fk = f" -> {col['foreign_key']}" if col.get('foreign_key') else ""
                    context_parts.append(f"  - {col['name']} {col['type']}{pk}{fk}: {col['description']}")
        
        # KPI 정보
        context_parts.append("\n=== 관련 KPI 공식 ===")
        for doc, score in results["kpi"]:
            if score > 0.1:
                context_parts.append(f"\n[{doc.metadata['kpi_name_kr']}] (유사도: {score:.3f})")
                context_parts.append(f"공식: {doc.metadata['formula']}")
                context_parts.append(f"관련 테이블: {', '.join(doc.metadata['related_tables'])}")
                if doc.metadata.get('sql_example'):
                    context_parts.append(f"예시 SQL:\n{doc.metadata['sql_example']}")
        
        # Golden SQL
        context_parts.append("\n=== 유사 Golden SQL ===")
        for doc, score in results["golden_sql"]:
            if score > 0.1:
                context_parts.append(f"\n[{doc.metadata['query_id']}] (유사도: {score:.3f})")
                context_parts.append(f"자연어: {doc.metadata['natural_language']}")
                context_parts.append(f"SQL:\n{doc.metadata['sql']}")
                context_parts.append(f"설명: {doc.metadata['explanation']}")
        
        return "\n".join(context_parts)


def demo():
    """데모 실행"""
    print("=" * 60)
    print("제조업 RAG 시스템 구축")
    print("=" * 60)
    
    # RAG 시스템 초기화
    rag_system = ManufacturingRAGSystem()
    
    # 데이터 로드
    print("\n[1] 데이터 로드 중...")
    rag_system.load_schema_data("./input/RAG/schema_Info.json")
    rag_system.load_kpi_data("./input/RAG/kpi.json")
    rag_system.load_golden_sql_data("./input/RAG/goldenSQL.json")
    
    # 테스트 쿼리들
    test_queries = [
        "이번 달 라인별 OEE 현황을 보여줘",
        "불량률이 높은 제품 TOP 5 알려줘",
        "설비 가동률과 MTBF 현황",
        "재고가 부족한 자재 목록",
        "월별 생산달성률 추이"
    ]
    
    print("\n" + "=" * 60)
    print("[2] RAG 검색 테스트")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"📝 사용자 질문: {query}")
        print('─' * 60)
        
        # 컨텍스트 생성
        context = rag_system.generate_context(query, top_k=2)
        print(context)
        
    return rag_system


# if __name__ == "__main__":
#     rag_system = demo()