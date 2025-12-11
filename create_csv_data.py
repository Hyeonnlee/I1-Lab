"""
Production Orders CSV 데이터 생성기
SQL 쿼리에 맞는 가상 데이터셋 생성
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import os


def generate_production_orders_csv(months=6, filename="production_orders.csv"):
    """
    Production Orders 테이블 CSV 생성
    
    테이블 스키마:
    - order_id: 주문 ID
    - production_date: 생산 날짜
    - line_id: 라인 ID (A, B, C, D)
    - product_id: 제품 ID
    - order_status: 주문 상태 (completed, in_progress, cancelled)
    - actual_quantity: 실제 생산 수량
    - defect_quantity: 불량 수량
    - target_quantity: 목표 수량
    - actual_start_time: 실제 시작 시간
    - actual_end_time: 실제 종료 시간
    - shift: 교대 (DAY, NIGHT)
    """
    
    print("=" * 80)
    print("🏭 Production Orders CSV 생성기")
    print("=" * 80)
    
    # 데이터 생성 설정
    lines = ['A', 'B', 'C', 'D']
    products = ['PROD-001', 'PROD-002', 'PROD-003', 'PROD-004', 'PROD-005']
    statuses = ['completed', 'in_progress', 'cancelled']
    status_weights = [0.85, 0.10, 0.05]  # completed 85%, in_progress 10%, cancelled 5%
    shifts = ['DAY', 'NIGHT']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)
    
    print(f"\n생성 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"생성 개월: {months}개월")
    
    # 데이터 리스트
    data = []
    order_counter = 1
    
    current_date = start_date
    while current_date <= end_date:
        # 하루에 라인별로 2-4개 주문 생성
        for line in lines:
            num_orders = random.randint(2, 4)
            
            for _ in range(num_orders):
                # 주문 상태 결정
                order_status = random.choices(statuses, weights=status_weights)[0]
                
                # 기본 정보
                order_id = f"ORD-{current_date.strftime('%Y%m%d')}-{order_counter:04d}"
                product_id = random.choice(products)
                shift = random.choice(shifts)
                
                # 수량 정보
                target_quantity = random.randint(800, 1200)
                
                if order_status == 'completed':
                    # 완료된 주문: 목표 대비 85~105% 생산
                    actual_quantity = int(target_quantity * random.uniform(0.85, 1.05))
                    # 불량률: 1~5%
                    defect_quantity = int(actual_quantity * random.uniform(0.01, 0.05))
                    
                    # 시작/종료 시간 설정
                    if shift == 'DAY':
                        start_hour = random.randint(8, 10)
                        duration_hours = random.uniform(6, 10)
                    else:
                        start_hour = random.randint(20, 22)
                        duration_hours = random.uniform(6, 10)
                    
                    actual_start = current_date.replace(hour=start_hour, minute=random.randint(0, 59))
                    actual_end = actual_start + timedelta(hours=duration_hours)
                    
                elif order_status == 'in_progress':
                    # 진행 중: 목표 대비 30~70% 생산
                    actual_quantity = int(target_quantity * random.uniform(0.30, 0.70))
                    defect_quantity = int(actual_quantity * random.uniform(0.01, 0.05))
                    
                    # 시작 시간만 있음
                    if shift == 'DAY':
                        start_hour = random.randint(8, 12)
                    else:
                        start_hour = random.randint(20, 23)
                    
                    actual_start = current_date.replace(hour=start_hour, minute=random.randint(0, 59))
                    actual_end = None  # 아직 종료 안됨
                    
                else:  # cancelled
                    # 취소된 주문
                    actual_quantity = 0
                    defect_quantity = 0
                    actual_start = None
                    actual_end = None
                
                # 데이터 추가
                data.append({
                    'order_id': order_id,
                    'production_date': current_date.strftime('%Y-%m-%d'),
                    'line_id': line,
                    'product_id': product_id,
                    'order_status': order_status,
                    'actual_quantity': actual_quantity,
                    'defect_quantity': defect_quantity,
                    'target_quantity': target_quantity,
                    'actual_start_time': actual_start.strftime('%Y-%m-%d %H:%M:%S') if actual_start else None,
                    'actual_end_time': actual_end.strftime('%Y-%m-%d %H:%M:%S') if actual_end else None,
                    'shift': shift
                })
                
                order_counter += 1
        
        current_date += timedelta(days=1)
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 통계 출력
    print(f"\n" + "=" * 80)
    print("📊 생성된 데이터 통계")
    print("=" * 80)
    print(f"총 레코드 수: {len(df):,}건")
    print(f"\n라인별 분포:")
    print(df['line_id'].value_counts().sort_index())
    print(f"\n주문 상태별 분포:")
    print(df['order_status'].value_counts())
    print(f"\n제품별 분포:")
    print(df['product_id'].value_counts().sort_index())
    
    # 완료된 주문에 대한 통계
    completed_df = df[df['order_status'] == 'completed']
    if len(completed_df) > 0:
        print(f"\n완료된 주문 분석:")
        print(f"  - 완료 건수: {len(completed_df):,}건")
        print(f"  - 평균 생산량: {completed_df['actual_quantity'].mean():.0f}개")
        print(f"  - 평균 불량률: {(completed_df['defect_quantity'].sum() / completed_df['actual_quantity'].sum() * 100):.2f}%")
    
    # A라인 완료 주문 통계 (쿼리와 관련)
    line_a_completed = df[(df['line_id'] == 'A') & (df['order_status'] == 'completed')]
    if len(line_a_completed) > 0:
        print(f"\nA라인 완료 주문 (쿼리 대상):")
        print(f"  - 건수: {len(line_a_completed):,}건")
        print(f"  - 총 생산량: {line_a_completed['actual_quantity'].sum():,}개")
        print(f"  - 총 불량량: {line_a_completed['defect_quantity'].sum():,}개")
        print(f"  - 불량률: {(line_a_completed['defect_quantity'].sum() / line_a_completed['actual_quantity'].sum() * 100):.2f}%")
    
    # 최근 3개월 데이터 통계
    three_months_ago = end_date - timedelta(days=90)
    recent_df = df[pd.to_datetime(df['production_date']) >= three_months_ago]
    recent_a_completed = recent_df[(recent_df['line_id'] == 'A') & (recent_df['order_status'] == 'completed')]
    
    if len(recent_a_completed) > 0:
        print(f"\n최근 3개월 A라인 완료 주문:")
        print(f"  - 건수: {len(recent_a_completed):,}건")
        print(f"  - 불량률: {(recent_a_completed['defect_quantity'].sum() / recent_a_completed['actual_quantity'].sum() * 100):.2f}%")
    
    # CSV 파일 저장
    output_dir = "sample_querydata"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    print(f"\n" + "=" * 80)
    print("✅ CSV 파일 생성 완료")
    print("=" * 80)
    print(f"파일 경로: {filepath}")
    print(f"파일 크기: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    # 데이터 미리보기
    print(f"\n" + "=" * 80)
    print("📋 데이터 미리보기 (상위 10건)")
    print("=" * 80)
    print(df.head(10).to_string(index=False))
    
    # SQL 쿼리 예시
    print(f"\n" + "=" * 80)
    print("💡 SQL 쿼리 예시")
    print("=" * 80)
    print("""
-- 원본 쿼리 (MySQL)
SELECT
    (SUM(defect_quantity) / SUM(actual_quantity)) * 100 as defect_rate
FROM production_orders
WHERE production_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH) 
    AND order_status = 'completed'
    AND line_id = 'A';

-- SQLite 버전
SELECT
    (SUM(defect_quantity) * 100.0 / SUM(actual_quantity)) as defect_rate
FROM production_orders
WHERE production_date >= date('now', '-3 months')
    AND order_status = 'completed'
    AND line_id = 'A';

-- 월별 불량률 추세
SELECT
    strftime('%Y-%m', production_date) as month,
    line_id,
    (SUM(defect_quantity) * 100.0 / SUM(actual_quantity)) as defect_rate,
    COUNT(*) as order_count
FROM production_orders
WHERE order_status = 'completed'
    AND production_date >= date('now', '-3 months')
GROUP BY strftime('%Y-%m', production_date), line_id
ORDER BY month, line_id;
    """)
    
    return df


def main():
    """메인 실행 함수"""
    
    # CSV 생성
    df = generate_production_orders_csv(months=6, filename="production_orders.csv")
    
    print("\n" + "=" * 80)
    print("🎉 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  - sample_data/production_orders.csv")
    print("\n다음 단계:")
    print("  1. CSV 파일을 데이터베이스에 import")
    print("  2. 제공된 SQL 쿼리 실행")
    print("  3. 결과 분석")
    

if __name__ == "__main__":
    main()