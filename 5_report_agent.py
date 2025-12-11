"""
제조 현장 분석 결과 리포트 생성 에이전트
Hugging Face Inference Client 기반
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_KEY")


class ReportGeneratorAgent:
    """분석 결과를 기반으로 종합 리포트를 생성하는 에이전트"""
    
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.client = InferenceClient(token=api_token)
        self.model_id = model_id
        
    def load_analysis_results(self, json_path: str) -> Dict[str, str]:
        """분석 결과 JSON 파일 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_executive_summary(self, results: Dict[str, str]) -> str:
        """경영진 요약 보고서 생성"""
        prompt = f"""당신은 제조 현장의 경영진을 위한 리포트 작성 전문가입니다.
**중요: 반드시 한국어로만 작성해주세요.**

다음은 4개의 AI 에이전트가 분석한 제조 현장 데이터 결과입니다:

=== 현황 분석 ===
{results['descriptive']}

=== 근본 원인 분석 ===
{results['diagnostic']}

=== 미래 예측 ===
{results['predictive']}

=== 최적화 방안 ===
{results['prescriptive']}

위 분석 결과를 바탕으로 **경영진을 위한 요약 보고서**를 작성해주세요:

1. 핵심 요약 (3-5문장으로 전체 상황 요약)
2. 주요 발견 사항 (TOP 3)
3. 비즈니스 영향도 분석
4. 즉시 실행 필요 액션 아이템 (우선순위 순)
5. 예상 효과 및 ROI

경영진이 빠르게 이해하고 의사결정할 수 있도록 명확하고 간결하게 한국어로 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1024,
            temperature=0.4,
        )
        return response.choices[0].message.content
    
    def generate_technical_report(self, results: Dict[str, str]) -> str:
        """기술팀을 위한 상세 리포트 생성"""
        prompt = f"""당신은 제조 현장의 기술팀을 위한 상세 리포트 작성 전문가입니다.
**중요: 반드시 한국어로만 작성해주세요.**

다음은 AI 에이전트의 분석 결과입니다:

=== 현황 분석 ===
{results['descriptive']}

=== 근본 원인 분석 ===
{results['diagnostic']}

=== 미래 예측 ===
{results['predictive']}

=== 최적화 방안 ===
{results['prescriptive']}

위 분석 결과를 바탕으로 **기술팀을 위한 상세 실행 계획**을 작성해주세요:

1. 문제 상황 상세 분석
2. 근본 원인별 기술적 해결 방안
3. 라인별/설비별 구체적 조치 사항
4. 단계별 실행 계획 (단기/중기/장기)
5. 필요한 기술 자원 및 인력
6. 리스크 및 제약 사항
7. 모니터링 지표 및 성공 기준

기술팀이 바로 실행에 옮길 수 있도록 구체적이고 실용적으로 한국어로 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1536,
            temperature=0.4,
        )
        return response.choices[0].message.content
    
    def generate_action_plan(self, results: Dict[str, str]) -> str:
        """액션 플랜 생성"""
        prompt = f"""당신은 제조 현장의 개선 프로젝트 매니저입니다.
**중요: 반드시 한국어로만 작성해주세요.**

다음은 AI 분석 결과입니다:

=== 최적화 방안 ===
{results['prescriptive']}

=== 예측 분석 ===
{results['predictive']}

위 분석을 바탕으로 **30일 액션 플랜**을 작성해주세요:

## Week 1 (긴급 조치)
- 액션 아이템
- 담당자/팀
- 완료 조건
- 예상 효과

## Week 2-3 (단기 개선)
- 액션 아이템
- 담당자/팀
- 완료 조건
- 예상 효과

## Week 4 (평가 및 조정)
- 액션 아이템
- 담당자/팀
- 완료 조건
- 예상 효과

각 주차별로 구체적이고 실행 가능한 계획을 한국어로 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1024,
            temperature=0.5,
        )
        return response.choices[0].message.content
    
    def generate_markdown_report(self, results: Dict[str, str]) -> str:
        """Markdown 형식의 종합 리포트 생성"""
        
        print("\n[1/4] 경영진 요약 보고서 생성 중...")
        executive_summary = self.generate_executive_summary(results)
        print("✓ 완료")
        
        print("\n[2/4] 기술팀 상세 리포트 생성 중...")
        technical_report = self.generate_technical_report(results)
        print("✓ 완료")
        
        print("\n[3/4] 액션 플랜 생성 중...")
        action_plan = self.generate_action_plan(results)
        print("✓ 완료")
        
        print("\n[4/4] 최종 리포트 조합 중...")
        
        # Markdown 리포트 생성
        report_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        markdown_report = f"""# 제조 현장 AI 분석 종합 리포트

**생성일**: {report_date}
**분석 시스템**: Multi-Agent AI 분석 시스템 (4개 에이전트)

---

## 📋 목차

1. [경영진 요약 보고서](#경영진-요약-보고서)
2. [원본 AI 분석 결과](#원본-ai-분석-결과)
3. [기술팀 상세 실행 계획](#기술팀-상세-실행-계획)
4. [30일 액션 플랜](#30일-액션-플랜)

---

## 📊 경영진 요약 보고서

{executive_summary}

---

## 🤖 원본 AI 분석 결과

### 1️⃣ 현황 분석 (설명 에이전트)

{results['descriptive']}

### 2️⃣ 근본 원인 분석 (진단 에이전트)

{results['diagnostic']}

### 3️⃣ 미래 예측 (예측 에이전트)

{results['predictive']}

### 4️⃣ 최적화 방안 (처방 에이전트)

{results['prescriptive']}

---

## 🔧 기술팀 상세 실행 계획

{technical_report}

---

## 📅 30일 액션 플랜

{action_plan}

---

## 📎 부록

### 생성 정보
- **분석 시스템**: 4-Agent 협업 시스템
  - 설명 에이전트 (Descriptive Agent)
  - 진단 에이전트 (Diagnostic Agent)
  - 근본 원인 분석 (RCA)
  - 예측 에이전트 (Predictive Agent)
  - 처방 에이전트 (Prescriptive Agent)
- **AI 모델**: {self.model_id}
- **생성 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### 리포트 사용 가이드
1. **경영진**: 첫 번째 섹션(요약 보고서)을 중심으로 검토
2. **기술팀**: 상세 실행 계획 섹션을 중심으로 실행
3. **프로젝트 매니저**: 액션 플랜을 기반으로 일정 관리
4. **전체 팀**: 원본 AI 분석 결과를 참고하여 상세 이해

---

**Report Generated by Manufacturing AI Agent System**
"""
        
        print("✓ 완료")
        return markdown_report
    
    def save_report(self, report: str, output_path: str):
        """리포트를 파일로 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ 리포트가 '{output_path}'에 저장되었습니다.")
    
    def generate_html_report(self, markdown_report: str) -> str:
        """Markdown을 HTML로 변환"""
        try:
            import markdown
            
            html_template = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>제조 현장 AI 분석 리포트</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {markdown.markdown(markdown_report, extensions=['tables', 'fenced_code'])}
    </div>
</body>
</html>"""
            return html_template
        except ImportError:
            print("⚠️  markdown 패키지가 설치되지 않았습니다. HTML 변환을 건너뜁니다.")
            print("   설치: pip install markdown")
            return None


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제조 현장 AI 분석 리포트 생성 시스템")
    print("=" * 80)
    
    # 리포트 생성기 초기화
    report_generator = ReportGeneratorAgent()
    
    # 분석 결과 로드
    analysis_results_path = "output/analysis_results.json"
    
    if not os.path.exists(analysis_results_path):
        print(f"\n❌ 오류: '{analysis_results_path}' 파일을 찾을 수 없습니다.")
        print("먼저 '4_analysis_agent.py'를 실행하여 분석 결과를 생성해주세요.")
        return
    
    print(f"\n📂 분석 결과 로드 중: {analysis_results_path}")
    results = report_generator.load_analysis_results(analysis_results_path)
    print("✓ 완료")
    
    # Markdown 리포트 생성
    print("\n" + "=" * 80)
    print("종합 리포트 생성 중...")
    print("=" * 80)
    
    markdown_report = report_generator.generate_markdown_report(results)
    
    # Markdown 파일 저장
    markdown_output_path = "output/comprehensive_report.md"
    report_generator.save_report(markdown_report, markdown_output_path)
    
    # HTML 리포트 생성 (선택사항)
    html_report = report_generator.generate_html_report(markdown_report)
    if html_report:
        html_output_path = "output/comprehensive_report.html"
        report_generator.save_report(html_report, html_output_path)
        print(f"✅ HTML 리포트가 '{html_output_path}'에 저장되었습니다.")
    
    print("\n" + "=" * 80)
    print("🎉 리포트 생성 완료!")
    print("=" * 80)
    print(f"\n생성된 파일:")
    print(f"  - Markdown: {markdown_output_path}")
    if html_report:
        print(f"  - HTML: {html_output_path}")
    print("\n리포트를 확인하세요!")


if __name__ == "__main__":
    main()