import os
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import sys
import io
from pathlib import Path

# 표준 출력(stdout)을 UTF-8로 인코딩하도록 재설정
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', line_buffering=True)
    except:
        pass

load_dotenv()

class DomainGlossaryLoader:
    """도메인 용어집을 다양한 형식으로 로드하는 클래스"""
    
    def __init__(self, glossary_path: Optional[str] = None):
        self.glossary_path = glossary_path or self._find_default_glossary()
        self.glossary: Dict[str, str] = {}
        
    def _find_default_glossary(self) -> str:
        """기본 용어집 파일 찾기"""
        default_paths = [
            "input/domain_glossary.json",
            "input/domain_glossary.txt",
            "config/domain_glossary.json",
            "domain_glossary.json",
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                return path
        
        return "input/domain_glossary.json"
    
    def load(self) -> Dict[str, str]:
        """용어집 로드"""
        if not os.path.exists(self.glossary_path):
            print(f"⚠ 용어집 파일이 없습니다: {self.glossary_path}")
            print(f"📝 기본 용어집을 생성합니다...")
            self._create_default_glossary()
        
        file_extension = Path(self.glossary_path).suffix.lower()
        
        try:
            if file_extension == '.json':
                self.glossary = self._load_json()
            elif file_extension == '.txt':
                self.glossary = self._load_txt()
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_extension}")
            
            print(f"✓ 용어집 로드 완료: {len(self.glossary)}개 항목")
            return self.glossary
            
        except Exception as e:
            print(f"✗ 용어집 로드 실패: {e}")
            print(f"📝 기본 용어집을 사용합니다.")
            return self._get_default_glossary()
    
    def _load_json(self) -> Dict[str, str]:
        """JSON 형식 로드"""
        with open(self.glossary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 중첩 구조 지원
            if "terms" in data:
                return self._flatten_dict(data["terms"])
            return data
    
    def _load_txt(self) -> Dict[str, str]:
        """TXT 형식 로드 (형식: 한국어 = 영문)"""
        glossary = {}
        with open(self.glossary_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    parts = line.split('=', 1)
                elif ':' in line:
                    parts = line.split(':', 1)
                elif '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    continue
                
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    glossary[key] = value
        
        return glossary
    
    def _flatten_dict(self, d: dict) -> Dict[str, str]:
        """중첩된 딕셔너리 평탄화"""
        items = {}
        for k, v in d.items():
            if isinstance(v, dict):
                items.update(self._flatten_dict(v))
            else:
                items[k] = v
        return items
    
    def _get_default_glossary(self) -> Dict[str, str]:
        """기본 용어집"""
        return {
            "수율": "YIELD",
            "3호기": "EQUIPMENT_M03",
            "1호기": "EQUIPMENT_M01",
            "2호기": "EQUIPMENT_M02",
            "지난주": "LAST_WEEK",
            "이번주": "THIS_WEEK",
            "왜 떨어졌어": "ANOMALY_ANALYSIS",
            "알려줘": "DATA_RETRIEVAL",
            "보여줘": "DATA_RETRIEVAL",
            "생산량": "PRODUCTION_VOLUME",
            "불량률": "DEFECT_RATE"
        }
    
    def _create_default_glossary(self):
        """기본 용어집 파일 생성"""
        os.makedirs(os.path.dirname(self.glossary_path) or '.', exist_ok=True)
        
        file_extension = Path(self.glossary_path).suffix.lower()
        
        if file_extension == '.json':
            self._save_json()
        elif file_extension == '.txt':
            self._save_txt()
        
        print(f"✓ 기본 용어집 생성: {self.glossary_path}")
    
    def _save_json(self):
        """JSON 형식으로 저장"""
        categorized = {
            "metadata": {
                "version": "1.0",
                "description": "제조 현장 도메인 용어집",
                "last_updated": datetime.now().isoformat()
            },
            "terms": {
                "metrics": {
                    "수율": "YIELD",
                    "생산량": "PRODUCTION_VOLUME",
                    "불량률": "DEFECT_RATE"
                },
                "equipment": {
                    "1호기": "EQUIPMENT_M01",
                    "2호기": "EQUIPMENT_M02",
                    "3호기": "EQUIPMENT_M03"
                },
                "time_expressions": {
                    "지난주": "LAST_WEEK",
                    "이번주": "THIS_WEEK"
                },
                "intents": {
                    "왜 떨어졌어": "ANOMALY_ANALYSIS",
                    "알려줘": "DATA_RETRIEVAL",
                    "보여줘": "DATA_RETRIEVAL"
                }
            }
        }
        
        with open(self.glossary_path, 'w', encoding='utf-8') as f:
            json.dump(categorized, f, indent=2, ensure_ascii=False)
    
    def _save_txt(self):
        """TXT 형식으로 저장"""
        default = self._get_default_glossary()
        
        with open(self.glossary_path, 'w', encoding='utf-8') as f:
            f.write("# 제조 현장 도메인 용어집\n")
            f.write("# 형식: 한국어_용어 = 영문_코드\n")
            f.write(f"# 최종 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 카테고리별로 정리
            f.write("# === 지표 (Metrics) ===\n")
            f.write("수율 = YIELD\n")
            f.write("생산량 = PRODUCTION_VOLUME\n")
            f.write("불량률 = DEFECT_RATE\n\n")
            
            f.write("# === 설비 (Equipment) ===\n")
            f.write("1호기 = EQUIPMENT_M01\n")
            f.write("2호기 = EQUIPMENT_M02\n")
            f.write("3호기 = EQUIPMENT_M03\n\n")
            
            f.write("# === 시간 표현 (Time) ===\n")
            f.write("지난주 = LAST_WEEK\n")
            f.write("이번주 = THIS_WEEK\n\n")
            
            f.write("# === 의도 (Intents) ===\n")
            f.write("왜 떨어졌어 = ANOMALY_ANALYSIS\n")
            f.write("알려줘 = DATA_RETRIEVAL\n")
            f.write("보여줘 = DATA_RETRIEVAL\n")


class QueryNormalizationAgent:
    def __init__(self, 
                 model_id: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 use_chat: bool = True,
                 glossary_path: Optional[str] = None):
        """
        Args:
            model_id: 사용할 LLM 모델 ID
            use_chat: Chat API 사용 여부
            glossary_path: 도메인 용어집 파일 경로
        """
        self.api_token = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_token:
            raise ValueError("HUGGINGFACE_API_KEY가 설정되지 않았습니다.")
        
        self.model_id = model_id
        self.use_chat = use_chat
        
        # Hugging Face Inference Client 초기화
        try:
            self.client = InferenceClient(token=self.api_token)
            print(f"✓ 클라이언트 초기화 성공")
            print(f"✓ 사용 모델: {model_id}")
            print(f"✓ API 방식: {'Chat Completion' if use_chat else 'Text Generation'}")
        except Exception as e:
            print(f"✗ 클라이언트 초기화 실패: {e}")
            raise
        
        # 도메인 용어집 로드
        print(f"\n{'='*50}")
        print("📚 도메인 용어집 로딩")
        print(f"{'='*50}")
        
        glossary_loader = DomainGlossaryLoader(glossary_path)
        self.domain_glossary = glossary_loader.load()
        
        # 용어집 내용 출력
        print("\n📋 로드된 용어집:")
        for i, (key, value) in enumerate(self.domain_glossary.items(), 1):
            print(f"  {i:2}. {key:15} → {value}")
        print(f"{'='*50}\n")

    def _preprocess_query(self, query: str) -> str:
        """도메인 용어집 기반 1차 정규화"""
        processed_query = query
        for k, v in self.domain_glossary.items():
            processed_query = processed_query.replace(k, v)
        return processed_query

    def _construct_system_message(self) -> str:
        """시스템 메시지 생성"""
        return """You are a Manufacturing Query Normalization Agent.
Convert user queries into JSON objects with these exact fields: intent, metric, time_frame, filter.

Rules:
1. 'intent' must be: ANOMALY_ANALYSIS or DATA_RETRIEVAL
2. 'metric' examples: YIELD, PRODUCTION_VOLUME, DEFECT_RATE
3. 'time_frame' has 'type' (RELATIVE or ABSOLUTE) and 'value'
4. 'filter' has 'field' and 'value'
5. Return ONLY valid JSON without any explanation or markdown

Examples:
Input: "지난주 3호기 수율이 왜 떨어졌어?"
Output: {"intent": "ANOMALY_ANALYSIS", "metric": "YIELD", "time_frame": {"type": "RELATIVE", "value": "LAST_WEEK"}, "filter": {"field": "EQUIPMENT_ID", "value": "EQUIPMENT_M03"}}

Input: "이번주 1호기 생산량 알려줘"
Output: {"intent": "DATA_RETRIEVAL", "metric": "PRODUCTION_VOLUME", "time_frame": {"type": "RELATIVE", "value": "THIS_WEEK"}, "filter": {"field": "EQUIPMENT_ID", "value": "EQUIPMENT_M01"}}"""

    def _construct_user_message(self, original_query: str) -> str:
        """사용자 메시지 생성"""
        return f'Convert this query: "{original_query}"'

    def _call_chat_completion(self, original_query: str) -> str:
        """Chat Completion API 호출"""
        messages = [
            {"role": "system", "content": self._construct_system_message()},
            {"role": "user", "content": self._construct_user_message(original_query)}
        ]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=300,
            temperature=0.1
        )
        
        return response.choices[0].message.content

    def _call_text_generation(self, original_query: str) -> str:
        """Text Generation API 호출"""
        prompt = f"""{self._construct_system_message()}

{self._construct_user_message(original_query)}

Output:"""
        
        response = self.client.text_generation(
            prompt,
            model=self.model_id,
            max_new_tokens=300,
            temperature=0.1,
            return_full_text=False
        )
        
        return response

    def _clean_json_output(self, text: str) -> str:
        """LLM 출력에서 JSON 추출"""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if json_match:
            json_str = json_match.group(0)
            json_str = json_str.replace("'", '"')
            return json_str
        
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            json_str = json_str.replace("'", '"')
            return json_str
        
        return text

    def normalize(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """메인 실행 함수"""
        if verbose:
            print(f"\n📝 입력 쿼리: {query}")
        
        pre_processed = self._preprocess_query(query)
        if verbose:
            print(f"🔄 전처리 완료: {pre_processed}")
        
        try:
            if verbose:
                print(f"🤖 LLM 호출 중...")
            
            if self.use_chat:
                response = self._call_chat_completion(query)
            else:
                response = self._call_text_generation(query)
            
            if verbose:
                print(f"✓ LLM 응답 받음")
            
        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"✗ LLM API 호출 실패: {error_msg}")
            
            if self.use_chat and "not supported" in error_msg.lower():
                if verbose:
                    print("🔄 Text Generation으로 재시도...")
                try:
                    response = self._call_text_generation(query)
                    if verbose:
                        print(f"✓ 재시도 성공!")
                except Exception as retry_error:
                    return {
                        "error": "All API methods failed",
                        "details": str(retry_error)
                    }
            else:
                return {"error": "LLM API Call Failed", "details": error_msg}

        cleaned_response = self._clean_json_output(response)
        
        try:
            result_json = json.loads(cleaned_response)
            if verbose:
                print("✓ JSON 파싱 성공")
            return result_json
        except json.JSONDecodeError as e:
            if verbose:
                print(f"✗ JSON 파싱 실패: {e}")
            return {
                "error": "Failed to parse JSON",
                "raw_output": response[:500]
            }

    def save_results_to_json(self, results: List[Dict[str, Any]], output_dir: str = "output") -> str:
        """정규화 결과를 JSON 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_normalization_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_id,
                "api_method": "chat_completion" if self.use_chat else "text_generation",
                "total_queries": len(results),
                "glossary_terms": len(self.domain_glossary)
            },
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장 완료: {filepath}")
        print(f"📁 파일 크기: {os.path.getsize(filepath)} bytes")
        
        return filepath


# --- 실행 ---
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Query Normalization Agent 시작")
    print("=" * 50)
    
    model_options = [
        ("mistralai/Mistral-7B-Instruct-v0.2", True),
        ("google/flan-t5-large", False),
    ]
    
    agent = None
    
    for model_id, use_chat in model_options:
        try:
            print(f"\n🔍 {model_id} 시도 중...")
            agent = QueryNormalizationAgent(
                model_id=model_id, 
                use_chat=use_chat,
                glossary_path="input/domain_glossary.json"  # 또는 .txt
            )
            break
        except Exception as e:
            print(f"✗ {model_id} 실패: {e}")
            continue
    
    if agent is None:
        print("\n❌ 사용 가능한 모델을 찾지 못했습니다.")
        exit(1)
    
    print(f"\n{'='*50}")
    print("💬 사용자 입력 모드")
    print("  - 질문 입력 후 Enter")
    print("  - 세미콜론(;)으로 여러 질문 구분")
    print("  - 종료: q, quit, exit")
    print(f"{'='*50}\n")
    
    all_results = []
    
    while True:
        try:
            user_input = input("📌 질문을 입력하세요 (종료: q): ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\n👋 프로그램을 종료합니다.")
                break
            
            if not user_input:
                print("⚠ 질문을 입력해주세요.\n")
                continue
            
            queries = [q.strip() for q in user_input.split(';') if q.strip()]
            
            for i, query in enumerate(queries, start=len(all_results) + 1):
                print(f"\n{'='*50}")
                print(f"테스트 케이스 {i}")
                print(f"{'='*50}")
                
                result = agent.normalize(query, verbose=True)
                
                result_with_query = {
                    "query_id": i,
                    "original_query": query,
                    "normalized_result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
                all_results.append(result_with_query)
                
                print(f"\n📊 최종 결과:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            
            print(f"\n현재까지 {len(all_results)}개의 질문이 처리되었습니다.")
            
        except KeyboardInterrupt:
            print("\n\n⚠ Ctrl+C 감지. 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            continue
    
    if all_results:
        saved_path = agent.save_results_to_json(all_results)
        
        print(f"\n{'='*50}")
        print("✅ 모든 작업 완료!")
        print(f"📂 결과 파일: {saved_path}")
        print(f"📊 총 처리: {len(all_results)}개")
        print(f"{'='*50}")
    else:
        print("\n⚠ 처리된 질문이 없습니다.")