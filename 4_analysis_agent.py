"""
제조 현장 데이터 분석 Multi-Agent 시스템
Hugging Face Hub + LangChain 기반
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# LangChain imports
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Hugging Face 설정
import os
from dotenv import load_dotenv

load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_KEY")

class ManufacturingDataAnalyzer:
    """제조 데이터 분석을 위한 기본 클래스"""
    
    def __init__(self, production_data: pd.DataFrame, qualitative_data: Dict):
        self.production_df = production_data
        self.qualitative_data = qualitative_data
        
    def get_summary_stats(self) -> Dict:
        """생산 데이터 요약 통계"""
        return {
            "total_orders": len(self.production_df),
            "completed_orders": len(self.production_df[self.production_df['order_status'] == 'completed']),
            "avg_defect_rate": (self.production_df['defect_quantity'] / 
                               self.production_df['actual_quantity']).mean() * 100,
            "lines": self.production_df['line_id'].unique().tolist(),
            "products": self.production_df['product_id'].unique().tolist()
        }
    
    def get_defect_analysis(self, line_id: Optional[str] = None) -> Dict:
        """불량률 분석"""
        df = self.production_df[self.production_df['order_status'] == 'completed'].copy()
        if line_id:
            df = df[df['line_id'] == line_id]
        
        df['defect_rate'] = (df['defect_quantity'] / df['actual_quantity']) * 100
        
        return {
            "average_defect_rate": df['defect_rate'].mean(),
            "max_defect_rate": df['defect_rate'].max(),
            "by_line": df.groupby('line_id')['defect_rate'].mean().to_dict(),
            "by_product": df.groupby('product_id')['defect_rate'].mean().to_dict(),
            "by_shift": df.groupby('shift')['defect_rate'].mean().to_dict()
        }
    
    def get_production_efficiency(self) -> Dict:
        """생산 효율 분석"""
        df = self.production_df[self.production_df['order_status'] == 'completed'].copy()
        df['achievement_rate'] = (df['actual_quantity'] / df['target_quantity']) * 100
        
        return {
            "avg_achievement_rate": df['achievement_rate'].mean(),
            "by_line": df.groupby('line_id')['achievement_rate'].mean().to_dict(),
            "underperforming_orders": len(df[df['achievement_rate'] < 90])
        }
    
    def find_correlations(self, threshold: float = 0.3) -> Dict:
        """변수 간 상관관계 분석"""
        df = self.production_df[self.production_df['order_status'] == 'completed'].copy()
        df['defect_rate'] = (df['defect_quantity'] / df['actual_quantity']) * 100
        df['achievement_rate'] = (df['actual_quantity'] / df['target_quantity']) * 100
        
        # 수치형 컬럼만 선택
        numeric_cols = ['actual_quantity', 'defect_quantity', 'target_quantity', 
                       'defect_rate', 'achievement_rate']
        corr_matrix = df[numeric_cols].corr()
        
        # 높은 상관관계 찾기
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j]
                    })
        
        return {"high_correlations": high_corr}
    
    def get_rag_context(self, anomaly_type: str) -> List[Dict]:
        """정성 데이터에서 관련 맥락 검색 (RAG)"""
        relevant_docs = []
        
        # 작업 일보에서 관련 문서 찾기
        for log in self.qualitative_data.get('work_logs', []):
            if any(tag in log.get('tags', []) for tag in 
                   ['생산량감소', '품질이상', '불량률증가', '설비이상']):
                relevant_docs.append({
                    "type": "work_log",
                    "content": log['content'],
                    "date": log['date'],
                    "line": log['line']
                })
        
        # 정비 로그에서 관련 문서 찾기
        for log in self.qualitative_data.get('maintenance_logs', []):
            relevant_docs.append({
                "type": "maintenance_log",
                "content": log['content'],
                "date": log['date'],
                "equipment": log['equipment']
            })
        
        return relevant_docs[:5]  # 최근 5개만 반환


def create_llm():
    """공통 LLM 생성 함수"""
    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=api_token,
        temperature=0.5,
        max_new_tokens=1024,
    )
    return ChatHuggingFace(llm=endpoint)


class DescriptiveAgent:
    """설명 에이전트: 현황 파악 및 데이터 요약"""
    
    def __init__(self, analyzer: ManufacturingDataAnalyzer):
        self.analyzer = analyzer
        self.llm = create_llm()
        
    def analyze(self) -> str:
        """현황 분석 수행"""
        stats = self.analyzer.get_summary_stats()
        defects = self.analyzer.get_defect_analysis()
        efficiency = self.analyzer.get_production_efficiency()
        
        prompt = f"""당신은 제조 현장 데이터 분석 전문가입니다. 다음 생산 데이터를 분석하여 현황을 요약해주세요.
**중요: 반드시 한국어로만 답변해주세요.**

전체 통계:
- 총 주문 수: {stats['total_orders']}
- 완료된 주문: {stats['completed_orders']}
- 평균 불량률: {stats['avg_defect_rate']:.2f}%
- 생산 라인: {', '.join(stats['lines'])}

불량 분석:
- 전체 평균 불량률: {defects['average_defect_rate']:.2f}%
- 최대 불량률: {defects['max_defect_rate']:.2f}%
- 라인별 불량률: {json.dumps(defects['by_line'], ensure_ascii=False)}
- 교대조별 불량률: {json.dumps(defects['by_shift'], ensure_ascii=False)}

생산 효율:
- 평균 목표 달성률: {efficiency['avg_achievement_rate']:.2f}%
- 저성과 주문 수: {efficiency['underperforming_orders']}

위 데이터를 바탕으로:
1. 전체적인 생산 현황 요약
2. 주요 문제점 식별
3. 개선이 필요한 영역 지적

간결하고 명확하게 한국어로 답변해주세요."""

        response = self.llm.invoke(prompt)
        # ChatHuggingFace는 메시지 객체를 반환하므로 content 추출
        return response.content if hasattr(response, 'content') else str(response)


class DiagnosticAgent:
    """진단 에이전트: 문제 원인 분석 (RCA)"""
    
    def __init__(self, analyzer: ManufacturingDataAnalyzer):
        self.analyzer = analyzer
        self.llm = create_llm()
        
    def analyze(self) -> str:
        """근본 원인 분석"""
        correlations = self.analyzer.find_correlations()
        defects = self.analyzer.get_defect_analysis()
        rag_context = self.analyzer.get_rag_context("defect")
        
        # RAG 컨텍스트 포맷팅
        context_str = "\n".join([
            f"[{doc['type']}] {doc['date']}: {doc['content'][:200]}..." 
            for doc in rag_context
        ])
        
        prompt = f"""당신은 제조 현장의 근본 원인 분석(RCA) 전문가입니다.
**중요: 반드시 한국어로만 답변해주세요.**

정량 데이터 분석:
- 라인별 불량률: {json.dumps(defects['by_line'], ensure_ascii=False)}
- 제품별 불량률: {json.dumps(defects['by_product'], ensure_ascii=False)}
- 교대조별 불량률: {json.dumps(defects['by_shift'], ensure_ascii=False)}
- 주요 상관관계: {json.dumps(correlations['high_correlations'][:3], ensure_ascii=False)}

정성 데이터 (작업 일지 및 정비 로그):
{context_str}

위 정량/정성 데이터를 종합하여:
1. 불량률이 높은 주요 원인 3가지 도출
2. 각 원인에 대한 근거 제시 (데이터 기반)
3. 라인별/제품별 특이사항 분석
4. 교대조별 차이가 나는 이유 추론

근거를 명확히 제시하며 한국어로 분석해주세요."""

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)


class PredictiveAgent:
    """예측 에이전트: 미래 예측"""
    
    def __init__(self, analyzer: ManufacturingDataAnalyzer):
        self.analyzer = analyzer
        self.llm = create_llm()
        
    def predict_defects(self) -> Dict:
        """불량률 예측"""
        df = self.analyzer.production_df[
            self.analyzer.production_df['order_status'] == 'completed'
        ].copy()
        df['defect_rate'] = (df['defect_quantity'] / df['actual_quantity']) * 100
        
        # 간단한 이동평균 기반 예측
        recent_data = df.tail(100)
        predictions = {}
        
        for line in df['line_id'].unique():
            line_data = recent_data[recent_data['line_id'] == line]
            if len(line_data) > 0:
                trend = line_data['defect_rate'].rolling(window=10).mean().iloc[-1]
                predictions[line] = {
                    "predicted_defect_rate": trend,
                    "risk_level": "높음" if trend > 4 else "중간" if trend > 2 else "낮음"
                }
        
        return predictions
    
    def analyze(self) -> str:
        """예측 분석 수행"""
        predictions = self.predict_defects()
        efficiency = self.analyzer.get_production_efficiency()
        
        prompt = f"""당신은 제조 현장의 예측 분석 전문가입니다.
**중요: 반드시 한국어로만 답변해주세요.**

과거 데이터 기반 예측:
- 라인별 예측 불량률: {json.dumps(predictions, ensure_ascii=False)}
- 현재 평균 목표 달성률: {efficiency['avg_achievement_rate']:.2f}%

다음 사항을 예측해주세요:
1. 향후 1주일간 각 라인의 불량률 추세
2. 설비 고장 가능성이 높은 라인 식별
3. 생산 목표 미달성 위험이 있는 제품/라인
4. 재고 최적화를 위한 생산량 조정 필요 여부

데이터 기반으로 구체적인 예측을 한국어로 제시해주세요."""

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)


class PrescriptiveAgent:
    """처방 에이전트: 최적 행동 제시"""
    
    def __init__(self, analyzer: ManufacturingDataAnalyzer):
        self.analyzer = analyzer
        self.llm = create_llm()
        
    def analyze(self, diagnostic_result: str, predictive_result: str) -> str:
        """처방 분석 수행"""
        defects = self.analyzer.get_defect_analysis()
        efficiency = self.analyzer.get_production_efficiency()
        
        prompt = f"""당신은 제조 현장의 최적화 전문가입니다.
**중요: 반드시 한국어로만 답변해주세요.**

진단 결과:
{diagnostic_result}

예측 결과:
{predictive_result}

현재 상태:
- 평균 불량률: {defects['average_defect_rate']:.2f}%
- 평균 목표 달성률: {efficiency['avg_achievement_rate']:.2f}%
- 저성과 주문: {efficiency['underperforming_orders']}건

위 분석을 바탕으로 다음 최적화 방안을 제시해주세요:

1. 공정 최적화:
   - 각 라인별 최적 설정값 제안
   - 불량률 감소를 위한 구체적 조치
   
2. 스케줄링 최적화:
   - 교대조별 작업 배분 개선안
   - 제품별 생산 순서 조정 제안
   
3. 자원 배분:
   - 정비 우선순위 설정
   - 인력 재배치 필요 여부
   
4. 즉시 실행 가능한 액션 아이템 (우선순위 순):

각 제안에 대해 예상 효과와 실행 방법을 한국어로 구체적으로 제시해주세요."""

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)


class ManufacturingAgentOrchestrator:
    """Multi-Agent 오케스트레이터"""
    
    def __init__(self, production_csv_path: str, qualitative_json_path: str):
        # 데이터 로드
        self.production_df = pd.read_csv(production_csv_path)
        
        with open(qualitative_json_path, 'r', encoding='utf-8') as f:
            self.qualitative_data = json.load(f)
        
        # 분석기 초기화
        self.analyzer = ManufacturingDataAnalyzer(
            self.production_df, 
            self.qualitative_data
        )
        
        # 에이전트 초기화
        self.descriptive_agent = DescriptiveAgent(self.analyzer)
        self.diagnostic_agent = DiagnosticAgent(self.analyzer)
        self.predictive_agent = PredictiveAgent(self.analyzer)
        self.prescriptive_agent = PrescriptiveAgent(self.analyzer)
        
    def run_analysis(self) -> Dict[str, str]:
        """전체 분석 파이프라인 실행"""
        
        print("=" * 80)
        print("제조 현장 AI 에이전트 분석 시작")
        print("=" * 80)
        
        # 1. 설명 에이전트
        print("\n[1/4] 설명 에이전트 실행 중...")
        descriptive_result = self.descriptive_agent.analyze()
        print("✓ 완료")
        
        # 2. 진단 에이전트
        print("\n[2/4] 진단 에이전트 (RCA) 실행 중...")
        diagnostic_result = self.diagnostic_agent.analyze()
        print("✓ 완료")
        
        # 3. 예측 에이전트
        print("\n[3/4] 예측 에이전트 실행 중...")
        predictive_result = self.predictive_agent.analyze()
        print("✓ 완료")
        
        # 4. 처방 에이전트
        print("\n[4/4] 처방 에이전트 실행 중...")
        prescriptive_result = self.prescriptive_agent.analyze(
            diagnostic_result, 
            predictive_result
        )
        print("✓ 완료")
        
        results = {
            "descriptive": descriptive_result,
            "diagnostic": diagnostic_result,
            "predictive": predictive_result,
            "prescriptive": prescriptive_result
        }
        
        return results
    
    def print_results(self, results: Dict[str, str]):
        """결과 출력"""
        print("\n" + "=" * 80)
        print("분석 결과 요약")
        print("=" * 80)
        
        print("\n📊 1. 현황 분석 (설명 에이전트)")
        print("-" * 80)
        print(results['descriptive'])
        
        print("\n🔍 2. 근본 원인 분석 (진단 에이전트)")
        print("-" * 80)
        print(results['diagnostic'])
        
        print("\n📈 3. 미래 예측 (예측 에이전트)")
        print("-" * 80)
        print(results['predictive'])
        
        print("\n💡 4. 최적화 방안 (처방 에이전트)")
        print("-" * 80)
        print(results['prescriptive'])


# 사용 예시
if __name__ == "__main__":
    # 오케스트레이터 초기화
    orchestrator = ManufacturingAgentOrchestrator(
        production_csv_path="input/sample_querydata/production_orders.csv",
        qualitative_json_path="input/RAG/qualitative_log.json"
    )
    
    # 분석 실행
    results = orchestrator.run_analysis()
    
    # 결과 출력
    orchestrator.print_results(results)
    
    # 결과 저장
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    with open('output/analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 분석 완료! 결과가 'analysis_results.json'에 저장되었습니다.")