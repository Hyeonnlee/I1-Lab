"""
Hugging Face 무료 임베딩 모델 기반 RAG SQL Generator
ChromaDB + Hugging Face Embedding Models + Inference API
"""

import requests
import json
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import hashlib
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


@dataclass
class TableSchema:
    """테이블 스키마 정의"""
    name: str
    columns: List[str]
    description: str
    sample_queries: List[str] = None


@dataclass
class KPIDefinition:
    """KPI 정의"""
    name: str
    formula: str
    description: str
    unit: str
    table: str
    related_terms: List[str] = None


@dataclass
class GoldenSQL:
    """검증된 SQL 예제"""
    question: str
    sql: str
    explanation: str
    tags: List[str]


class HuggingFaceEmbedding:
    """Hugging Face 임베딩 모델 래퍼"""
    
    # 추천 무료 임베딩 모델 목록
    EMBEDDING_MODELS = {
        'bge_small_en': {
            'name': 'BAAI/bge-small-en-v1.5',
            'dimension': 384,
            'description': '가볍고 빠른 영어 임베딩 (추천)',
            'language': 'English'
        },
        'bge_base_en': {
            'name': 'BAAI/bge-base-en-v1.5',
            'dimension': 768,
            'description': '균형잡힌 영어 임베딩',
            'language': 'English'
        },
        'bge_large_en': {
            'name': 'BAAI/bge-large-en-v1.5',
            'dimension': 1024,
            'description': '고성능 영어 임베딩',
            'language': 'English'
        },
        'bge_m3': {
            'name': 'BAAI/bge-m3',
            'dimension': 1024,
            'description': '다국어 임베딩 (한국어 포함)',
            'language': 'Multilingual'
        },
        'multilingual_e5_small': {
            'name': 'intfloat/multilingual-e5-small',
            'dimension': 384,
            'description': '다국어 소형 모델 (한국어 지원)',
            'language': 'Multilingual'
        },
        'multilingual_e5_base': {
            'name': 'intfloat/multilingual-e5-base',
            'dimension': 768,
            'description': '다국어 기본 모델 (한국어 우수)',
            'language': 'Multilingual'
        },
        'multilingual_e5_large': {
            'name': 'intfloat/multilingual-e5-large',
            'dimension': 1024,
            'description': '다국어 대형 모델 (최고 성능)',
            'language': 'Multilingual'
        },
        'gte_small': {
            'name': 'thenlper/gte-small',
            'dimension': 384,
            'description': '경량 범용 임베딩',
            'language': 'English'
        },
        'gte_base': {
            'name': 'thenlper/gte-base',
            'dimension': 768,
            'description': '범용 임베딩',
            'language': 'English'
        },
        'all_minilm': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'description': '초경량 빠른 임베딩',
            'language': 'English'
        },
        'paraphrase_multilingual': {
            'name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'dimension': 384,
            'description': '다국어 경량 임베딩',
            'language': 'Multilingual'
        }
    }
    
    def __init__(self, model_key: str = 'multilingual_e5_base', hf_token: str = None):
        """
        임베딩 모델 초기화
        
        Args:
            model_key: EMBEDDING_MODELS의 키
            hf_token: Hugging Face API 토큰
        """
        self.model_key = model_key
        self.model_info = self.EMBEDDING_MODELS.get(model_key)
        
        if not self.model_info:
            raise ValueError(f"지원하지 않는 모델: {model_key}")
        
        self.model_name = self.model_info['name']
        self.dimension = self.model_info['dimension']
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_API_KEY")
        
        print(f"\n🎯 임베딩 모델: {self.model_name}")
        print(f"   - 설명: {self.model_info['description']}")
        print(f"   - 언어: {self.model_info['language']}")
        print(f"   - 차원: {self.dimension}D")
    
    @classmethod
    def list_models(cls):
        """사용 가능한 모델 목록 출력"""
        print("\n" + "="*70)
        print("📋 사용 가능한 임베딩 모델")
        print("="*70)
        
        for key, info in cls.EMBEDDING_MODELS.items():
            lang_emoji = "🌍" if info['language'] == 'Multilingual' else "🇬🇧"
            print(f"\n{lang_emoji} [{key}]")
            print(f"   모델: {info['name']}")
            print(f"   설명: {info['description']}")
            print(f"   차원: {info['dimension']}D")
        
        print("\n" + "="*70)
        print("💡 추천:")
        print("   - 한국어 사용: multilingual_e5_base (균형)")
        print("   - 빠른 속도: multilingual_e5_small (가벼움)")
        print("   - 최고 품질: multilingual_e5_large (느림)")
        print("   - 영어만: bge_small_en (최고 속도)")
        print("="*70 + "\n")
    
    def create_embedding_function(self):
        """ChromaDB용 임베딩 함수 생성"""
        # 로컬 모델 사용 (sentence-transformers) - 더 안정적
        try:
            from sentence_transformers import SentenceTransformer
            
            class LocalEmbeddingFunction:
                def __init__(self, model_name):
                    print(f"  📥 로컬 임베딩 모델 다운로드 중... (처음만 시간 소요)")
                    try:
                        self.model = SentenceTransformer(model_name)
                        print(f"  ✓ 모델 로딩 완료: {model_name}")
                    except Exception as e:
                        print(f"  ❌ 모델 로딩 실패: {e}")
                        raise
                
                def __call__(self, input):
                    if isinstance(input, str):
                        input = [input]
                    try:
                        embeddings = self.model.encode(input, convert_to_numpy=True)
                        return embeddings.tolist()
                    except Exception as e:
                        print(f"  ❌ 임베딩 생성 실패: {e}")
                        raise
            
            return LocalEmbeddingFunction(self.model_name)
        
        except ImportError:
            print("\n" + "="*70)
            print("❌ sentence-transformers가 설치되지 않았습니다!")
            print("="*70)
            print("\n설치 방법:")
            print("  pip install sentence-transformers torch")
            print("\n또는 conda 사용:")
            print("  conda install -c conda-forge sentence-transformers")
            print("\n" + "="*70)
            raise ImportError("sentence-transformers 패키지가 필요합니다")


class VectorDBManager:
    """벡터 DB 관리자 (임베딩 모델 선택 가능)"""
    
    def __init__(self, 
                 embedding_model: str = 'multilingual_e5_base',
                 hf_token: str = None,
                 persist_directory: str = "./chroma_db_hf"):
        """
        초기화
        
        Args:
            embedding_model: 임베딩 모델 키
            hf_token: Hugging Face 토큰
            persist_directory: 저장 경로
        """
        self.embedding_model_key = embedding_model
        self.persist_directory = persist_directory
        
        # 임베딩 모델 초기화
        self.embedding = HuggingFaceEmbedding(embedding_model, hf_token)
        self.embedding_function = self.embedding.create_embedding_function()
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 컬렉션 생성 (임베딩 함수 적용)
        self.metadata_collection = self._get_or_create_collection(
            "metadata_rag", self.embedding_function
        )
        self.business_collection = self._get_or_create_collection(
            "business_logic_rag", self.embedding_function
        )
        self.fewshot_collection = self._get_or_create_collection(
            "fewshot_sql_rag", self.embedding_function
        )
        
        print(f"✓ ChromaDB 초기화 완료 (저장 경로: {persist_directory})")
    
    def _get_or_create_collection(self, name: str, embedding_function):
        """컬렉션 가져오기 또는 생성"""
        try:
            return self.client.get_collection(
                name=name,
                embedding_function=embedding_function
            )
        except:
            return self.client.create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
    
    def reset_all(self):
        """모든 컬렉션 초기화"""
        try:
            self.client.delete_collection("metadata_rag")
            self.client.delete_collection("business_logic_rag")
            self.client.delete_collection("fewshot_sql_rag")
        except:
            pass
        
        self.metadata_collection = self._get_or_create_collection(
            "metadata_rag", self.embedding_function
        )
        self.business_collection = self._get_or_create_collection(
            "business_logic_rag", self.embedding_function
        )
        self.fewshot_collection = self._get_or_create_collection(
            "fewshot_sql_rag", self.embedding_function
        )
        
        print("✓ 모든 컬렉션이 초기화되었습니다")
    
    def add_metadata(self, tables: List[TableSchema]):
        """메타데이터 추가"""
        documents = []
        metadatas = []
        ids = []
        
        for table in tables:
            doc_text = f"""
테이블명: {table.name}
설명: {table.description}
컬럼: {', '.join(table.columns)}
샘플 쿼리: {', '.join(table.sample_queries or [])}
            """.strip()
            
            documents.append(doc_text)
            metadatas.append({
                'type': 'table_schema',
                'table_name': table.name,
                'description': table.description
            })
            ids.append(f"table_{table.name}")
        
        if documents:
            try:
                self.metadata_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✓ 메타데이터 {len(documents)}개 추가 완료")
            except Exception as e:
                print(f"❌ 메타데이터 추가 실패: {e}")
                raise
    
    def add_business_logic(self, kpis: Dict[str, KPIDefinition]):
        """비즈니스 로직 추가"""
        documents = []
        metadatas = []
        ids = []
        
        for key, kpi in kpis.items():
            related_terms_str = ', '.join(kpi.related_terms or [])
            doc_text = f"""
KPI명: {kpi.name}
메트릭: {key}
설명: {kpi.description}
공식: {kpi.formula}
단위: {kpi.unit}
관련 용어: {related_terms_str}
테이블: {kpi.table}
            """.strip()
            
            documents.append(doc_text)
            metadatas.append({
                'type': 'kpi_definition',
                'metric': key,
                'name': kpi.name,
                'formula': kpi.formula
            })
            ids.append(f"kpi_{key}")
        
        if documents:
            self.business_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✓ 비즈니스 로직 {len(documents)}개 추가 완료")
    
    def add_golden_sqls(self, sqls: List[GoldenSQL]):
        """Golden SQL 추가"""
        documents = []
        metadatas = []
        ids = []
        
        for sql_obj in sqls:
            doc_text = f"""
질문: {sql_obj.question}
설명: {sql_obj.explanation}
태그: {', '.join(sql_obj.tags)}
            """.strip()
            
            documents.append(doc_text)
            metadatas.append({
                'type': 'golden_sql',
                'question': sql_obj.question,
                'tags': ','.join(sql_obj.tags),
                'sql': sql_obj.sql
            })
            
            sql_id = hashlib.md5(sql_obj.question.encode()).hexdigest()
            ids.append(f"sql_{sql_id}")
        
        if documents:
            self.fewshot_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✓ Golden SQL {len(documents)}개 추가 완료")
    
    def search_metadata(self, query: str, top_k: int = 3) -> List[Dict]:
        """메타데이터 검색"""
        results = self.metadata_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return self._format_results(results)
    
    def search_business_logic(self, query: str, top_k: int = 3) -> List[Dict]:
        """비즈니스 로직 검색"""
        results = self.business_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return self._format_results(results)
    
    def search_golden_sqls(self, query: str, top_k: int = 3) -> List[Dict]:
        """Golden SQL 검색"""
        results = self.fewshot_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return self._format_results(results)
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """검색 결과 포맷팅"""
        formatted = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted
        
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted
    
    def get_collection_stats(self) -> Dict[str, int]:
        """컬렉션 통계"""
        return {
            'metadata': self.metadata_collection.count(),
            'business_logic': self.business_collection.count(),
            'golden_sql': self.fewshot_collection.count()
        }


class HuggingFaceSQLAgent:
    """Hugging Face 기반 SQL Generator (임베딩 모델 선택 가능)"""
    
    # LLM 모델 목록
    LLM_MODELS = {
        'qwen_coder_32b': 'Qwen/Qwen2.5-Coder-32B-Instruct',
        'deepseek_33b': 'deepseek-ai/deepseek-coder-33b-instruct',
        'codellama_34b': 'codellama/CodeLlama-34b-Instruct-hf',
        'mistral_7b': 'mistralai/Mistral-7B-Instruct-v0.3',
        'qwen_7b': 'Qwen/Qwen2.5-7B-Instruct',
        'phi3_medium': 'microsoft/Phi-3-medium-128k-instruct',
    }
    
    def __init__(self, 
                 hf_token: str = None,
                 embedding_model: str = 'multilingual_e5_base',
                 llm_model: str = 'mistral_7b'):
        """
        초기화
        
        Args:
            hf_token: Hugging Face API 토큰
            embedding_model: 임베딩 모델 키
            llm_model: LLM 모델 키
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        # LLM 모델 설정
        if llm_model in self.LLM_MODELS:
            self.llm_model_id = self.LLM_MODELS[llm_model]
        else:
            self.llm_model_id = llm_model
        
        print(f"\n🤖 LLM 모델: {self.llm_model_id}")
        
        # API 설정
        self.api_url = f"https://api-inference.huggingface.co/models/{self.llm_model_id}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        # 벡터 DB 초기화 (임베딩 모델 적용)
        print("\n" + "="*70)
        print("🔧 벡터 DB 초기화 중...")
        print("="*70)
        
        self.vector_db = VectorDBManager(
            embedding_model=embedding_model,
            hf_token=self.hf_token
        )
        
        # 데이터 초기화
        self._initialize_data()
        
        print("\n" + "="*70)
        print("✅ SQL Generator Agent 초기화 완료!")
        print("="*70)
        
        stats = self.vector_db.get_collection_stats()
        print(f"\n📊 벡터 DB 통계:")
        print(f"   - 메타데이터: {stats['metadata']}개")
        print(f"   - 비즈니스 로직: {stats['business_logic']}개")
        print(f"   - Golden SQL: {stats['golden_sql']}개\n")
    
    def _initialize_data(self):
        """RAG 데이터 초기화"""
        # 기존 데이터가 있으면 건너뛰기
        stats = self.vector_db.get_collection_stats()
        if stats['metadata'] > 0:
            print("✓ 기존 데이터 사용")
            return
        
        print("\n📥 RAG 데이터 초기화 중...")
        
        # 1. 테이블 스키마
        tables = [
            TableSchema(
                name='defect_records',
                columns=['record_id', 'line_id', 'defect_count', 'total_count', 
                        'defect_rate', 'record_date', 'shift', 'operator_id'],
                description='불량 기록 테이블. 일별/시프트별 불량률 데이터 저장',
                sample_queries=['불량률 추세', '불량 건수', '라인별 불량']
            ),
            TableSchema(
                name='process_variables',
                columns=['var_id', 'line_id', 'temperature', 'pressure', 
                        'speed', 'humidity', 'timestamp'],
                description='공정 변수 테이블. 온도, 압력, 속도 등 실시간 공정 데이터',
                sample_queries=['온도 데이터', '압력 추이', '공정 변수']
            ),
            TableSchema(
                name='production_lines',
                columns=['line_id', 'line_name', 'factory_id', 'status'],
                description='생산 라인 마스터 테이블',
                sample_queries=['라인 정보', '가동 라인']
            ),
            TableSchema(
                name='quality_events',
                columns=['event_id', 'line_id', 'event_type', 'severity', 
                        'description', 'occurred_at'],
                description='품질 이벤트 로그',
                sample_queries=['품질 이슈', '이상 발생']
            )
        ]
        self.vector_db.add_metadata(tables)
        
        # 2. KPI 정의
        kpis = {
            'DEFECT_RATE': KPIDefinition(
                name='불량률',
                formula='(defect_count / total_count) * 100',
                description='불량률 = (불량 수 / 전체 생산 수) × 100',
                unit='%',
                table='defect_records',
                related_terms=['불량', '품질', '결함', 'defect', 'quality']
            ),
            'CORRELATION': KPIDefinition(
                name='상관관계',
                formula='CORR(variable1, variable2)',
                description='두 변수 간의 상관계수 (-1 ~ 1)',
                unit='coefficient',
                table='process_variables',
                related_terms=['상관', 'correlation', '관계', '영향']
            )
        }
        self.vector_db.add_business_logic(kpis)
        
        # 3. Golden SQL
        golden_sqls = [
            GoldenSQL(
                question="지난 3개월간 불량률 추세 분석",
                sql="""SELECT 
    DATE_FORMAT(record_date, '%Y-%m') AS month,
    AVG(defect_rate) AS avg_defect_rate,
    MIN(defect_rate) AS min_defect_rate,
    MAX(defect_rate) AS max_defect_rate,
    STDDEV(defect_rate) AS stddev_defect_rate
FROM defect_records
WHERE line_id = 'A'
    AND record_date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)
GROUP BY DATE_FORMAT(record_date, '%Y-%m')
ORDER BY month;""",
                explanation="월별 불량률 통계 집계",
                tags=['trend', 'aggregation', 'defect_rate', 'monthly']
            ),
            GoldenSQL(
                question="불량률과 공정 변수 상관관계 분석",
                sql="""WITH defect_stats AS (
    SELECT 
        DATE(record_date) AS date,
        AVG(defect_rate) AS avg_defect_rate
    FROM defect_records
    WHERE line_id = 'A'
        AND record_date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)
    GROUP BY DATE(record_date)
),
process_stats AS (
    SELECT
        DATE(timestamp) AS date,
        AVG(temperature) AS avg_temp,
        AVG(pressure) AS avg_pressure
    FROM process_variables
    WHERE line_id = 'A'
        AND timestamp >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)
    GROUP BY DATE(timestamp)
)
SELECT
    'temperature' AS variable,
    CORR(d.avg_defect_rate, p.avg_temp) AS correlation
FROM defect_stats d
JOIN process_stats p ON d.date = p.date
UNION ALL
SELECT
    'pressure' AS variable,
    CORR(d.avg_defect_rate, p.avg_pressure) AS correlation
FROM defect_stats d
JOIN process_stats p ON d.date = p.date;""",
                explanation="CTE와 CORR 함수로 상관관계 분석",
                tags=['correlation', 'cte', 'join', 'causality']
            ),
            GoldenSQL(
                question="이동평균으로 불량률 이상치 탐지",
                sql="""WITH daily_metrics AS (
    SELECT
        record_date,
        defect_rate,
        AVG(defect_rate) OVER (
            ORDER BY record_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS moving_avg_7days,
        STDDEV(defect_rate) OVER (
            ORDER BY record_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS moving_stddev
    FROM defect_records
    WHERE line_id = 'A'
        AND record_date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)
)
SELECT
    record_date,
    defect_rate,
    moving_avg_7days,
    CASE
        WHEN defect_rate > moving_avg_7days + (2 * moving_stddev) THEN 'HIGH_ANOMALY'
        WHEN defect_rate < moving_avg_7days - (2 * moving_stddev) THEN 'LOW_ANOMALY'
        ELSE 'NORMAL'
    END AS anomaly_status
FROM daily_metrics
ORDER BY record_date;""",
                explanation="윈도우 함수로 이동평균 계산 후 2-sigma 기준 이상치 탐지",
                tags=['window_function', 'moving_average', 'anomaly_detection']
            )
        ]
        self.vector_db.add_golden_sqls(golden_sqls)
    
    def generate_sql(self, user_query: str, line_id: str = 'A') -> Dict[str, Any]:
        """자연어 → SQL 생성"""
        
        print(f"\n{'='*70}")
        print(f"📝 질문: {user_query}")
        print(f"🏭 라인: {line_id}")
        print(f"{'='*70}\n")
        
        # 1. RAG 검색
        print("🔍 유사 컨텍스트 검색 중...")
        
        print("  [1/3] 테이블 스키마 검색...")
        tables = self.vector_db.search_metadata(user_query, top_k=2)
        
        print("  [2/3] KPI 정의 검색...")
        kpis = self.vector_db.search_business_logic(user_query, top_k=2)
        
        print("  [3/3] Golden SQL 검색...")
        examples = self.vector_db.search_golden_sqls(user_query, top_k=2)
        
        print(f"  ✓ 총 {len(tables) + len(kpis) + len(examples)}개 컨텍스트 검색 완료\n")
        
        # 2. 프롬프트 구성
        prompt = self._build_prompt(user_query, line_id, tables, kpis, examples)
        
        # 3. LLM 호출
        print("🤖 SQL 생성 중...")
        response = self._generate_sql_with_llm(prompt)
        
        # 4. SQL 추출
        sql = self._extract_sql(response)
        
        print("✅ SQL 생성 완료!\n")
        
        return {
            'user_query': user_query,
            'line_id': line_id,
            'sql': sql,
            'raw_response': response,
            'rag_context': {
                'tables': tables,
                'kpis': kpis,
                'examples': examples
            }
        }
    
    def _build_prompt(self, query: str, line_id: str, 
                     tables: List[Dict], kpis: List[Dict], examples: List[Dict]) -> str:
        """프롬프트 구성"""
        
        # 스키마 정보
        schema_text = "## 데이터베이스 스키마\n\n"
        for t in tables:
            schema_text += f"### {t['metadata'].get('table_name', 'Unknown')}\n"
            schema_text += f"{t['document']}\n\n"
        
        # KPI 정보
        kpi_text = "## KPI 정의\n\n"
        for k in kpis:
            kpi_text += f"### {k['metadata'].get('name', 'KPI')}\n"
            kpi_text += f"{k['document']}\n\n"
        
        # 예제 SQL
        example_text = "## 참고 SQL 예제\n\n"
        for i, ex in enumerate(examples, 1):
            example_text += f"### 예제 {i}\n"
            example_text += f"```sql\n{ex['metadata'].get('sql', '')}\n```\n\n"
        
        prompt = f"""당신은 SQL 전문가입니다. 자연어 질문을 분석하여 정확한 SQL 쿼리를 생성하세요.

{schema_text}

{kpi_text}

{example_text}

## 요청
질문: {query}
대상 라인: {line_id}

위 정보를 참고하여 line_id = '{line_id}' 조건을 포함한 SQL 쿼리를 생성하세요.
반드시 ```sql 코드 블록 안에 SQL만 작성하세요.

```sql
"""
        
        return prompt
    
    def _generate_sql_with_llm(self, prompt: str) -> str:
        """LLM으로 SQL 생성"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            elif isinstance(result, dict):
                return result.get('generated_text', '')
            else:
                return str(result)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API 오류: {e}")
            return "-- SQL 생성 실패"
    
    def _extract_sql(self, response: str) -> str:
        """응답에서 SQL 추출"""
        # ```sql ... ``` 블록 찾기
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # ``` ... ``` 블록 찾기
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # 전체 응답 반환
        return response.strip()
    
    def print_result(self, result: Dict[str, Any]):
        """결과 출력"""
        print("="*70)
        print("📊 SQL 생성 결과")
        print("="*70)
        print(f"\n📝 질문: {result['user_query']}")
        print(f"🏭 라인: {result['line_id']}")
        
        print("\n" + "-"*70)
        print("💾 생성된 SQL:")
        print("-"*70)
        print(result['sql'])
        
        print("\n" + "-"*70)
        print("🔍 사용된 RAG 컨텍스트:")
        print("-"*70)
        ctx = result['rag_context']
        print(f"  - 테이블: {len(ctx['tables'])}개")
        print(f"  - KPI: {len(ctx['kpis'])}개")
        print(f"  - 예제 SQL: {len(ctx['examples'])}개")
        
        print("\n" + "="*70 + "\n")
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n" + "="*70)
        print("🚀 대화형 SQL Generator")
        print("="*70)
        print("\n명령어:")
        print("  - 질문 입력: SQL 생성")
        print("  - 'quit' / 'exit': 종료")
        print("  - 'reset': 벡터 DB 초기화")
        print("  - 'stats': 통계 보기")
        print("  - 'models': 임베딩 모델 목록")
        print()
        
        while True:
            try:
                query = input("\n💬 질문: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit']:
                    print("\n👋 종료합니다.\n")
                    break
                
                if query.lower() == 'reset':
                    self.vector_db.reset_all()
                    self._initialize_data()
                    print("✓ 벡터 DB 초기화 및 데이터 재로딩 완료")
                    continue
                
                if query.lower() == 'stats':
                    stats = self.vector_db.get_collection_stats()
                    print(f"\n📊 벡터 DB 통계:")
                    print(f"  - 메타데이터: {stats['metadata']}개")
                    print(f"  - 비즈니스 로직: {stats['business_logic']}개")
                    print(f"  - Golden SQL: {stats['golden_sql']}개")
                    continue
                
                if query.lower() == 'models':
                    HuggingFaceEmbedding.list_models()
                    continue
                
                line_id = input("🏭 라인 ID (기본: A): ").strip() or 'A'
                
                result = self.generate_sql(query, line_id)
                self.print_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 종료합니다.\n")
                break
            except Exception as e:
                print(f"\n❌ 오류: {str(e)}\n")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 함수"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  🤗 Hugging Face 무료 모델 기반 RAG SQL Generator               ║
║                                                                  ║
║  - 임베딩 모델: 11가지 무료 모델 선택 가능                      ║
║  - LLM 모델: 6가지 코드 생성 모델 선택 가능                     ║
║  - RAG: ChromaDB + 시맨틱 검색                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # HF 토큰
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("⚠️  HF_TOKEN 환경 변수가 설정되지 않았습니다.")
        print("   https://huggingface.co/settings/tokens 에서 무료 토큰 발급")
        print("   export HF_TOKEN='hf_...'")
        print()
        hf_token = input("HF 토큰 입력 (Enter=로컬 모델 사용): ").strip()
        if not hf_token:
            print("\n💡 로컬 임베딩 모델을 사용합니다 (sentence-transformers 필요)")
            print("   pip install sentence-transformers\n")
    
    # 임베딩 모델 선택
    print("\n" + "="*70)
    print("📋 임베딩 모델 선택")
    print("="*70)
    print("\n추천 모델:")
    print("1. multilingual_e5_base - 다국어 균형형 (추천) ⭐")
    print("2. multilingual_e5_small - 다국어 경량 (빠름)")
    print("3. multilingual_e5_large - 다국어 대형 (고성능)")
    print("4. bge_small_en - 영어 경량 (매우 빠름)")
    print("5. bge_m3 - 다국어 고성능")
    print("6. all - 전체 모델 목록 보기")
    
    emb_choice = input("\n임베딩 모델 선택 (1-6, 기본=1): ").strip() or '1'
    
    if emb_choice == '6':
        HuggingFaceEmbedding.list_models()
        emb_model = input("\n모델 키 입력: ").strip() or 'multilingual_e5_base'
    else:
        emb_map = {
            '1': 'multilingual_e5_base',
            '2': 'multilingual_e5_small',
            '3': 'multilingual_e5_large',
            '4': 'bge_small_en',
            '5': 'bge_m3'
        }
        emb_model = emb_map.get(emb_choice, 'multilingual_e5_base')
    
    # LLM 모델 선택
    print("\n" + "="*70)
    print("🤖 LLM 모델 선택")
    print("="*70)
    print("\n1. mistral_7b - 가볍고 빠름 (추천) ⭐")
    print("2. qwen_7b - 한국어 우수")
    print("3. qwen_coder_32b - 최고 성능 (느림)")
    print("4. deepseek_33b - 코드 생성 특화")
    
    llm_choice = input("\nLLM 모델 선택 (1-4, 기본=1): ").strip() or '1'
    
    llm_map = {
        '1': 'mistral_7b',
        '2': 'qwen_7b',
        '3': 'qwen_coder_32b',
        '4': 'deepseek_33b'
    }
    llm_model = llm_map.get(llm_choice, 'mistral_7b')
    
    # Agent 초기화
    print("\n" + "="*70)
    print("⚙️  Agent 초기화 중...")
    print("="*70)
    
    try:
        agent = HuggingFaceSQLAgent(
            hf_token=hf_token if hf_token else None,
            embedding_model=emb_model,
            llm_model=llm_model
        )
    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}")
        print("\n💡 해결 방법:")
        print("   1. HF 토큰 확인")
        print("   2. sentence-transformers 설치: pip install sentence-transformers")
        print("   3. 인터넷 연결 확인")
        return
    
    # 예제 테스트
    print("\n" + "="*70)
    print("🎯 예제 테스트")
    print("="*70)
    
    test_queries = [
        "지난 3개월간 A라인의 불량률 추세를 분석해줘",
        "불량률과 온도의 상관관계를 분석해줘"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[예제 {i}/{len(test_queries)}]")
        result = agent.generate_sql(query, line_id='A')
        agent.print_result(result)
        
        if i < len(test_queries):
            input("⏸️  다음 예제 (Enter)...")
    
    # 대화형 모드
    print("\n" + "="*70)
    user_input = input("대화형 모드로 전환? (y/n): ").strip().lower()
    
    if user_input == 'y':
        agent.interactive_mode()
    else:
        print("\n👋 종료합니다.\n")


if __name__ == "__main__":
    main()