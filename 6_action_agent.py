"""
액션 에이전트: 종합 리포트를 분석하여 사용자별 맞춤 조치 제안
Hugging Face Inference Client 기반
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_KEY")

class ActionAgent:
    """리포트를 읽고 사용자별 맞춤 액션을 제안하는 에이전트"""
    
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.client = InferenceClient(token=api_token)
        self.model_id = model_id
        
    def read_report(self, md_path: str) -> str:
        """Markdown 리포트 읽기 (더 정확한 원본 데이터)"""
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"리포트 파일을 찾을 수 없습니다: {md_path}")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_key_insights(self, report_content: str) -> str:
        """리포트에서 핵심 인사이트 추출"""
        prompt = f"""당신은 제조 현장 데이터 분석 전문가입니다.
**중요: 반드시 한국어로만 답변해주세요.**

다음은 AI가 생성한 제조 현장 종합 분석 리포트입니다:

{report_content[:8000]}  # 토큰 제한을 고려한 일부 추출

위 리포트를 분석하여 다음을 추출해주세요:

1. 가장 심각한 문제 3가지 (우선순위 순)
2. 각 문제의 영향도 (높음/중간/낮음)
3. 즉시 조치가 필요한 이슈
4. 1주일 내 해결 가능한 이슈
5. 장기 프로젝트가 필요한 이슈

간결하고 명확하게 한국어로 정리해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content
    
    def generate_ceo_actions(self, key_insights: str) -> str:
        """CEO/경영진을 위한 전략적 액션 제안"""
        prompt = f"""당신은 제조 기업의 경영 컨설턴트입니다.
**중요: 반드시 한국어로만 답변해주세요.**

다음은 현장 분석에서 추출한 핵심 인사이트입니다:

{key_insights}

CEO/경영진을 위한 **전략적 의사결정 액션**을 제안해주세요:

## 1. 긴급 의사결정 사항 (24-48시간 내)
- 결정 내용
- 예상 예산
- 기대 효과

## 2. 중기 전략 수정 (1개월 내)
- 조정 방향
- 필요 자원
- ROI 예측

## 3. 경영진 검토 필요 사항
- 검토 주제
- 배경 설명
- 옵션 제시

## 4. 리스크 관리
- 주요 리스크
- 대응 방안
- 모니터링 지표

경영진 관점에서 실행 가능하고 구체적으로 한국어로 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1536,
            temperature=0.4,
        )
        return response.choices[0].message.content
    
    def generate_manager_actions(self, key_insights: str) -> str:
        """생산 관리자를 위한 운영 액션 제안"""
        prompt = f"""당신은 제조 현장의 생산 관리 전문가입니다.
**중요: 반드시 한국어로만 답변해주세요.**

다음은 현장 분석 핵심 인사이트입니다:

{key_insights}

생산 관리자를 위한 **현장 운영 개선 액션**을 제안해주세요:

## 1. 오늘 할 일 (Today's Priority)
- 점검 항목
- 조치 사항
- 책임자 지정

## 2. 이번 주 개선 과제 (This Week)
- 개선 항목
- 실행 방법
- 목표 지표

## 3. 팀별 액션 아이템
- 생산팀
- 품질팀
- 정비팀
- 자재팀

## 4. 일일 모니터링 체크리스트
- 확인 항목
- 기준값
- 이상 발생 시 조치

현장에서 바로 실행 가능한 구체적인 내용으로 한국어로 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1536,
            temperature=0.4,
        )
        return response.choices[0].message.content
    
    def generate_engineer_actions(self, key_insights: str) -> str:
        """엔지니어/기술팀을 위한 기술적 액션 제안"""
        prompt = f"""당신은 제조 설비 및 공정 엔지니어입니다.
**중요: 반드시 한국어로만 답변해주세요.**

다음은 현장 분석 핵심 인사이트입니다:

{key_insights}

엔지니어/기술팀을 위한 **기술적 해결 액션**을 제안해주세요:

## 1. 긴급 설비 점검 대상
- 설비명
- 점검 항목
- 예상 소요 시간
- 필요 부품/도구

## 2. 공정 파라미터 조정
- 라인/설비
- 현재값 → 권장값
- 조정 근거
- 검증 방법

## 3. 예방 정비 계획
- 일정
- 대상 설비
- 정비 내용
- 필요 자원

## 4. 기술 개선 프로젝트
- 프로젝트명
- 목표
- 필요 기술/도구
- 예상 기간

엔지니어가 바로 작업에 착수할 수 있도록 기술적으로 구체적이고 한국어로 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1536,
            temperature=0.4,
        )
        return response.choices[0].message.content
    
    def generate_quality_actions(self, key_insights: str) -> str:
        """품질팀을 위한 품질 개선 액션 제안"""
        prompt = f"""당신은 제조 현장의 품질 관리 전문가입니다.
**중요: 반드시 한국어로만 답변해주세요.**

다음은 현장 분석 핵심 인사이트입니다:

{key_insights}

품질팀을 위한 **품질 개선 액션**을 제안해주세요:

## 1. 긴급 품질 점검 (Immediate)
- 점검 대상 (라인/제품)
- 불량 유형
- 샘플링 계획
- 판정 기준

## 2. 불량 저감 활동 (Short-term)
- 대상 불량 유형
- 원인 분석 방법
- 개선 방안
- 목표 불량률

## 3. 품질 시스템 강화 (Mid-term)
- 검사 기준 개선
- 교육 훈련 계획
- SOP 업데이트
- 측정 장비 교정

## 4. 데이터 기반 품질 관리
- 수집 지표
- 분석 방법
- 리포팅 주기
- 피드백 루프

품질팀이 실행할 수 있는 구체적이고 측정 가능한 내용으로 한국어로 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1536,
            temperature=0.4,
        )
        return response.choices[0].message.content
    
    def generate_interactive_actions(self, key_insights: str, user_role: str, 
                                    specific_concern: str = None) -> str:
        """사용자 맞춤형 대화형 액션 제안"""
        concern_text = f"\n\n특별히 관심있는 영역: {specific_concern}" if specific_concern else ""
        
        prompt = f"""당신은 제조 현장의 AI 어시스턴트입니다.
**중요: 반드시 한국어로만 답변해주세요.**

현장 분석 핵심 인사이트:
{key_insights}

사용자 역할: {user_role}{concern_text}

위 사용자의 역할과 관심사를 고려하여 **맞춤형 액션 플랜**을 제안해주세요:

## 1. 당신이 오늘 해야 할 일
- 구체적 액션 3-5개
- 각 액션의 중요도와 이유

## 2. 이번 주 중점 과제
- 주요 과제 2-3개
- 실행 방법과 체크포인트

## 3. 협업이 필요한 부분
- 누구와 협업할지
- 무엇을 논의할지
- 언제까지 완료할지

## 4. 성과 측정 방법
- 측정 지표
- 목표값
- 보고 방식

사용자의 입장에서 공감하며, 실행 가능하고 구체적으로 한국어로 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=1536,
            temperature=0.5,
        )
        return response.choices[0].message.content
    
    def create_action_dashboard(self, actions_dict: Dict[str, str]) -> str:
        """모든 액션을 통합한 대시보드 형식 리포트 생성"""
        
        timestamp = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
        today = datetime.now().strftime("%Y-%m-%d")
        week_end = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        dashboard = f"""# 🎯 제조 현장 액션 대시보드

**생성 일시**: {timestamp}  
**실행 기간**: {today} ~ {week_end}

---

## 📊 역할별 액션 플랜 요약

### 👔 경영진 (CEO/임원)
{actions_dict.get('ceo', 'N/A')}

---

### 👨‍💼 생산 관리자
{actions_dict.get('manager', 'N/A')}

---

### 🔧 엔지니어/기술팀
{actions_dict.get('engineer', 'N/A')}

---

### ✅ 품질팀
{actions_dict.get('quality', 'N/A')}

---

## 🔄 통합 실행 타임라인

### 즉시 (24시간 내)
- [ ] 경영진: 긴급 의사결정 사항 검토
- [ ] 생산 관리자: 오늘의 우선순위 실행
- [ ] 엔지니어: 긴급 설비 점검
- [ ] 품질팀: 긴급 품질 점검

### 단기 (1주일 내)
- [ ] 경영진: 중기 전략 수정 회의
- [ ] 생산 관리자: 주간 개선 과제 추진
- [ ] 엔지니어: 공정 파라미터 조정
- [ ] 품질팀: 불량 저감 활동

### 중기 (1개월 내)
- [ ] 경영진: 리스크 관리 시스템 구축
- [ ] 생산 관리자: 팀별 액션 아이템 완료
- [ ] 엔지니어: 예방 정비 계획 실행
- [ ] 품질팀: 품질 시스템 강화

---

## 📈 성과 지표 (KPI)

| 지표 | 현재 | 목표 (1개월) | 담당 |
|------|------|-------------|------|
| 불량률 | - | -2%p | 품질팀 |
| 생산 달성률 | - | +5%p | 생산팀 |
| 설비 가동률 | - | +3%p | 정비팀 |
| 비용 절감 | - | 목표 설정 | 전체 |

---

## ⚠️ 주의사항 및 리스크

1. **크로스 체크 필요**: 각 팀의 액션이 서로 충돌하지 않는지 확인
2. **일일 모니터링**: 진행 상황을 매일 체크
3. **주간 리뷰**: 매주 금요일 진행 상황 리뷰 회의
4. **유연한 조정**: 현장 상황에 따라 우선순위 조정 가능

---

## 📞 에스컬레이션 프로세스

```
현장 이슈 발견
    ↓
담당자 1차 대응 (30분 내)
    ↓
해결 안 되면 → 관리자 보고 (1시간 내)
    ↓
심각한 이슈 → 경영진 보고 (즉시)
```

---

**💡 Tip**: 이 대시보드를 인쇄하여 현장에 게시하거나, 매일 아침 미팅에서 활용하세요.

**Generated by Manufacturing Action Agent System**
"""
        return dashboard
    
    def save_actions(self, content: str, filepath: str):
        """액션 플랜 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 저장 완료: {filepath}")


def interactive_mode(agent: ActionAgent, key_insights: str):
    """대화형 모드: 사용자 입력을 받아 맞춤 액션 제공"""
    print("\n" + "=" * 80)
    print("🤖 대화형 액션 제안 모드")
    print("=" * 80)
    
    print("\n당신의 역할을 선택하세요:")
    print("1. CEO/경영진")
    print("2. 생산 관리자")
    print("3. 엔지니어/기술팀")
    print("4. 품질 관리자")
    print("5. 기타 (직접 입력)")
    
    role_map = {
        "1": "CEO/경영진",
        "2": "생산 관리자",
        "3": "엔지니어/기술팀",
        "4": "품질 관리자"
    }
    
    choice = input("\n선택 (1-5): ").strip()
    
    if choice in role_map:
        role = role_map[choice]
    elif choice == "5":
        role = input("역할을 입력하세요: ").strip()
    else:
        role = "현장 담당자"
    
    concern = input("\n특별히 관심있는 영역이 있다면 입력하세요 (없으면 Enter): ").strip()
    concern = concern if concern else None
    
    print(f"\n🔄 {role}님을 위한 맞춤 액션을 생성하고 있습니다...")
    
    custom_action = agent.generate_interactive_actions(key_insights, role, concern)
    
    print("\n" + "=" * 80)
    print(f"🎯 {role}님을 위한 맞춤 액션 플랜")
    print("=" * 80)
    print(custom_action)
    
    # 저장 여부 확인
    save = input("\n이 액션 플랜을 파일로 저장하시겠습니까? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"output/custom_action_{role.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        agent.save_actions(custom_action, filename)


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🎯 제조 현장 액션 에이전트")
    print("=" * 80)
    
    # 액션 에이전트 초기화
    agent = ActionAgent()
    
    # 리포트 파일 확인
    md_report_path = "output/comprehensive_report.md"
    
    if not os.path.exists(md_report_path):
        print(f"\n❌ 오류: '{md_report_path}' 파일을 찾을 수 없습니다.")
        print("먼저 '5_report_agent.py'를 실행하여 리포트를 생성해주세요.")
        return
    
    print(f"\n📂 리포트 읽기: {md_report_path}")
    report_content = agent.read_report(md_report_path)
    print("✓ 완료")
    
    # 핵심 인사이트 추출
    print("\n🔍 핵심 인사이트 추출 중...")
    key_insights = agent.extract_key_insights(report_content)
    print("✓ 완료")
    
    print("\n" + "=" * 80)
    print("📊 추출된 핵심 인사이트")
    print("=" * 80)
    print(key_insights)
    
    # 역할별 액션 생성
    print("\n" + "=" * 80)
    print("🎯 역할별 액션 플랜 생성 중...")
    print("=" * 80)
    
    actions_dict = {}
    
    print("\n[1/4] CEO/경영진 액션 생성 중...")
    actions_dict['ceo'] = agent.generate_ceo_actions(key_insights)
    print("✓ 완료")
    
    print("\n[2/4] 생산 관리자 액션 생성 중...")
    actions_dict['manager'] = agent.generate_manager_actions(key_insights)
    print("✓ 완료")
    
    print("\n[3/4] 엔지니어 액션 생성 중...")
    actions_dict['engineer'] = agent.generate_engineer_actions(key_insights)
    print("✓ 완료")
    
    print("\n[4/4] 품질팀 액션 생성 중...")
    actions_dict['quality'] = agent.generate_quality_actions(key_insights)
    print("✓ 완료")
    
    # 통합 대시보드 생성
    print("\n📊 통합 액션 대시보드 생성 중...")
    dashboard = agent.create_action_dashboard(actions_dict)
    
    # 저장
    dashboard_path = "output/action_dashboard.md"
    agent.save_actions(dashboard, dashboard_path)
    
    # 개별 액션 플랜도 저장
    for role, action in actions_dict.items():
        filepath = f"output/action_{role}.md"
        agent.save_actions(action, filepath)
    
    print("\n" + "=" * 80)
    print("🎉 액션 플랜 생성 완료!")
    print("=" * 80)
    print(f"\n생성된 파일:")
    print(f"  - 통합 대시보드: {dashboard_path}")
    print(f"  - CEO 액션: output/action_ceo.md")
    print(f"  - 관리자 액션: output/action_manager.md")
    print(f"  - 엔지니어 액션: output/action_engineer.md")
    print(f"  - 품질팀 액션: output/action_quality.md")
    
    # 대화형 모드 제안
    print("\n" + "=" * 80)
    interactive = input("\n💬 개인 맞춤형 액션을 받고 싶으신가요? (y/n): ").strip().lower()
    if interactive == 'y':
        interactive_mode(agent, key_insights)
    
    print("\n✨ 모든 작업 완료!")


if __name__ == "__main__":
    main()