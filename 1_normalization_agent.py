import os
import json
import re
from typing import Dict, Any, Optional, List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import sys
import io

# 표준 출력(stdout)을 UTF-8로 인코딩하도록 재설정
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', line_buffering=True)
    except:
        pass

load_dotenv()

class QueryNormalizationAgent:
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2", use_chat: bool = True):
        """
        개선사항:
        1. Chat Completion API 지원 추가
        2. 여러 모델 옵션 지원
        3. 자동 fallback 메커니즘
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
        
        # 도메인 용어집
        self.domain_glossary = {
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

    def _preprocess_query(self, query: str) -> str:
        """1단계: 도메인 용어집을 기반으로 텍스트를 1차 정규화"""
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
        """Text Generation API 호출 (fallback용)"""
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
        """LLM 출력에서 JSON 부분만 추출하고 정리"""
        try:
            # Markdown 코드 블록 제거
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            
            # JSON 패턴 찾기 (중첩된 객체 지원)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
            if json_match:
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"')
                return json_str
            
            # 백업: 첫 번째 { 부터 마지막 } 까지
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                json_str = json_str.replace("'", '"')
                return json_str
            
            return text
        except Exception as e:
            print(f"⚠ JSON 정리 중 에러: {e}")
            return text

    def normalize(self, query: str) -> Dict[str, Any]:
        """메인 실행 함수"""
        print(f"\n📝 입력 쿼리: {query}")
        
        # 1. 전처리
        pre_processed = self._preprocess_query(query)
        print(f"🔄 전처리 완료: {pre_processed}")
        
        # 2. LLM 호출
        try:
            print(f"🤖 LLM 호출 중... (모델: {self.model_id})")
            
            if self.use_chat:
                response = self._call_chat_completion(query)
            else:
                response = self._call_text_generation(query)
            
            print(f"✓ LLM 응답 받음 (길이: {len(response)} chars)")
            print(f"📄 원본 응답: {response[:200]}...")
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ LLM API 호출 실패: {error_msg}")
            
            # Chat에서 실패했으면 Text Generation으로 재시도
            if self.use_chat and "not supported" in error_msg.lower():
                print("🔄 Text Generation으로 재시도...")
                try:
                    response = self._call_text_generation(query)
                    print(f"✓ 재시도 성공!")
                except Exception as retry_error:
                    return {
                        "error": "All API methods failed",
                        "chat_error": error_msg,
                        "text_gen_error": str(retry_error),
                        "model": self.model_id
                    }
            else:
                return {
                    "error": "LLM API Call Failed",
                    "details": error_msg,
                    "model": self.model_id
                }

        # 3. 결과 파싱
        cleaned_response = self._clean_json_output(response)
        print(f"🧹 정리된 응답: {cleaned_response}")
        
        try:
            result_json = json.loads(cleaned_response)
            print("✓ JSON 파싱 성공")
            return result_json
        except json.JSONDecodeError as e:
            print(f"✗ JSON 파싱 실패: {e}")
            return {
                "error": "Failed to parse JSON",
                "parsing_error": str(e),
                "raw_output": response[:500],
                "cleaned_output": cleaned_response
            }

# --- 실행 테스트 ---
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Query Normalization Agent 시작")
    print("=" * 50)
    
    # 여러 모델 옵션 (우선순위대로)
    model_options = [
        ("mistralai/Mistral-7B-Instruct-v0.2", True),   # Chat API
        ("meta-llama/Llama-2-7b-chat-hf", True),        # Chat API
        ("google/flan-t5-large", False),                # Text Generation
        ("bigscience/bloom-1b7", False),                # Text Generation (fallback)
    ]
    
    agent = None
    
    # 사용 가능한 모델 찾기
    for model_id, use_chat in model_options:
        try:
            print(f"\n🔍 {model_id} 시도 중...")
            agent = QueryNormalizationAgent(model_id=model_id, use_chat=use_chat)
            break
        except Exception as e:
            print(f"✗ {model_id} 실패: {e}")
            continue
    
    if agent is None:
        print("\n❌ 사용 가능한 모델을 찾지 못했습니다.")
        print("💡 다음을 확인해주세요:")
        print("  1. HUGGINGFACE_API_KEY가 올바르게 설정되었는지")
        print("  2. 인터넷 연결 상태")
        print("  3. Hugging Face 서비스 상태: https://status.huggingface.co/")
        exit(1)
    
    # 테스트 케이스들
    test_queries = [
        "지난주 3호기 수율이 왜 떨어졌어?",
        "이번주 1호기 생산량 데이터 보여줘"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"테스트 케이스 {i}")
        print(f"{'='*50}")
        
        result = agent.normalize(query)
        
        print(f"\n📊 최종 결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))