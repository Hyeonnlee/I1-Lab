"""
RAG 기반 SQL 쿼리 생성기
output 폴더의 정규화된 JSON 데이터를 SQL 쿼리로 변환
"""

from rag_system import ManufacturingRAGSystem
from typing import Dict, List, Optional
import json
import os
from datetime import datetime


class SQLQueryGenerator:
    """RAG 기반 SQL 쿼리 생성기"""
    
    def __init__(self, rag_system: ManufacturingRAGSystem):
        self.rag = rag_system
        
    def parse_normalized_query(self, normalized_data: Dict) -> Dict:
        """정규화된 JSON 데이터를 SQL 생성용 의도로 변환"""
        intent = {
            "intent_type": normalized_data.get("intent"),
            "metric": normalized_data.get("metric"),
            "time_range": self._parse_time_range(normalized_data.get("time_frame", {})),
            "filter": normalized_data.get("filter"),
            "aggregation": self._infer_aggregation(normalized_data),
            "grouping": self._infer_grouping(normalized_data),
            "ordering": None,
            "limit": None,
            "additional_params": normalized_data.get("additional_params", {})
        }
        
        # ANOMALY_ANALYSIS의 경우 정렬 추가
        if intent["intent_type"] == "ANOMALY_ANALYSIS":
            intent["ordering"] = "DESC"
            intent["limit"] = 10
        
        return intent
    
    def _parse_time_range(self, time_frame: Dict) -> str:
        """시간 프레임을 SQL 조건으로 변환"""
        if not time_frame:
            return None
        
        time_type = time_frame.get("type")
        value = time_frame.get("value")
        
        if time_type == "ABSOLUTE":
            time_mappings = {
                "TODAY": "DATE(CURDATE())",
                "YESTERDAY": "DATE_SUB(CURDATE(), INTERVAL 1 DAY)",
                "THIS_WEEK": "YEARWEEK(CURDATE(), 1)",
                "THIS_MONTH": "DATE_FORMAT(CURDATE(), '%Y-%m')",
                "LAST_MONTH": "DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%Y-%m')",
                "LAST_3_MONTHS": "DATE_SUB(CURDATE(), INTERVAL 3 MONTH)",
                "LAST_6_MONTHS": "DATE_SUB(CURDATE(), INTERVAL 6 MONTH)",
                "THIS_YEAR": "YEAR(CURDATE())",
            }
            return time_mappings.get(value, "CURDATE()")
        
        elif time_type == "RELATIVE":
            unit = time_frame.get("unit", "DAY")
            count = time_frame.get("count", 1)
            return f"DATE_SUB(CURDATE(), INTERVAL {count} {unit})"
        
        elif time_type == "RANGE":
            start = time_frame.get("start")
            end = time_frame.get("end")
            return f"BETWEEN '{start}' AND '{end}'"
        
        return None
    
    def _infer_aggregation(self, normalized_data: Dict) -> str:
        """메트릭과 의도에서 집계 함수 추론"""
        intent = normalized_data.get("intent")
        metric = normalized_data.get("metric")
        
        # 의도에 따른 집계
        if intent in ["TREND_ANALYSIS", "COMPARISON"]:
            return "AVG"
        elif intent == "AGGREGATION":
            return "SUM"
        elif intent == "ANOMALY_ANALYSIS":
            return "MAX"
        
        # 메트릭에 따른 집계
        if metric in ["DEFECT_RATE", "YIELD_RATE", "OEE", "AVAILABILITY"]:
            return "AVG"
        elif metric in ["PRODUCTION_QUANTITY", "INVENTORY", "ENERGY"]:
            return "SUM"
        
        return None
    
    def _infer_grouping(self, normalized_data: Dict) -> str:
        """필터와 의도에서 그룹핑 추론"""
        intent = normalized_data.get("intent")
        filter_data = normalized_data.get("filter", {})
        
        # 필터가 있으면 해당 필드로 그룹핑
        if filter_data and filter_data.get("field"):
            field = filter_data.get("field")
            field_mappings = {
                "LINE_ID": "line_id",
                "PRODUCT_ID": "product_id",
                "EQUIPMENT_ID": "equipment_id",
                "FACTORY_ID": "factory_id",
                "DEPARTMENT_ID": "department_id",
                "SHIFT": "shift",
                "SUPPLIER_ID": "supplier_id",
            }
            return field_mappings.get(field, field.lower())
        
        # 추세 분석의 경우 시간별 그룹핑
        if intent == "TREND_ANALYSIS":
            time_frame = normalized_data.get("time_frame", {})
            if "MONTH" in str(time_frame.get("value", "")):
                return "DATE_FORMAT(production_date, '%Y-%m')"
            elif "WEEK" in str(time_frame.get("value", "")):
                return "YEARWEEK(production_date, 1)"
            else:
                return "DATE(production_date)"
        
        return None
    
    def generate_sql_from_json(self, json_data: Dict) -> Dict:
        """정규화된 JSON 데이터로부터 SQL 생성"""
        
        # 1. JSON에서 정규화된 쿼리 추출
        if "results" in json_data and len(json_data["results"]) > 0:
            query_result = json_data["results"][0]
            normalized_data = query_result.get("normalized_result", {})
            original_query = query_result.get("original_query", "")
        else:
            normalized_data = json_data
            original_query = ""
        
        # 2. 의도 파싱
        intent = self.parse_normalized_query(normalized_data)
        
        # 3. RAG 검색 (원본 쿼리 또는 메트릭으로)
        search_query = original_query if original_query else intent.get("metric", "")
        results = self.rag.search_all(search_query, top_k=3)
        
        # 4. Golden SQL에서 가장 유사한 쿼리 찾기
        best_golden_sql = None
        best_score = 0
        for doc, score in results['golden_sql']:
            if score > best_score:
                best_score = score
                best_golden_sql = doc.metadata
        
        # 5. 관련 스키마 정보 수집
        relevant_tables = []
        for doc, score in results['schema']:
            if score > 0.1:
                relevant_tables.append(doc.metadata)
        
        # 6. 관련 KPI 정보 수집
        relevant_kpis = []
        for doc, score in results['kpi']:
            if score > 0.1:
                relevant_kpis.append(doc.metadata)
        
        # 7. 메트릭 기반 KPI 매칭
        metric = intent.get("metric")
        if metric and not relevant_kpis:
            metric_to_kpi = {
                "DEFECT_RATE": "불량률",
                "YIELD_RATE": "수율",
                "OEE": "OEE",
                "AVAILABILITY": "가동률",
                "PRODUCTION_QUANTITY": "생산량",
                "THROUGHPUT": "처리량",
                "MTBF": "MTBF",
                "MTTR": "MTTR",
            }
            kpi_name = metric_to_kpi.get(metric)
            if kpi_name:
                kpi_results = self.rag.search_all(kpi_name, top_k=1)
                for doc, score in kpi_results['kpi']:
                    if score > 0.1:
                        relevant_kpis.append(doc.metadata)
        
        # 8. 결과 구성
        result = {
            "original_query": original_query,
            "normalized_data": normalized_data,
            "intent_analysis": intent,
            "relevant_tables": [t['table_name'] for t in relevant_tables],
            "relevant_kpis": [k['kpi_name_kr'] for k in relevant_kpis],
            "recommended_sql": None,
            "sql_source": None,
            "confidence": 0,
            "explanation": ""
        }
        
        # 9. SQL 생성 또는 추천
        if best_golden_sql and best_score > 0.3:
            # Golden SQL 활용
            result["recommended_sql"] = self._adapt_golden_sql(best_golden_sql['sql'], intent)
            result["sql_source"] = "golden_sql"
            result["confidence"] = min(best_score * 100, 95)
            result["explanation"] = f"Golden SQL '{best_golden_sql['query_id']}'를 기반으로 생성. {best_golden_sql['explanation']}"
        elif relevant_kpis and relevant_kpis[0].get('sql_example'):
            # KPI SQL 예시 활용
            result["recommended_sql"] = self._adapt_kpi_sql(relevant_kpis[0]['sql_example'], intent)
            result["sql_source"] = "kpi_example"
            result["confidence"] = 70
            result["explanation"] = f"KPI '{relevant_kpis[0]['kpi_name_kr']}' 계산 예시 SQL 활용"
        else:
            # 기본 쿼리 생성
            result["recommended_sql"] = self._generate_basic_sql(intent, relevant_tables, normalized_data)
            result["sql_source"] = "generated"
            result["confidence"] = 50
            result["explanation"] = "정규화된 데이터와 스키마 정보를 기반으로 기본 쿼리 생성"
        
        return result
    
    def _adapt_golden_sql(self, base_sql: str, intent: Dict) -> str:
        """Golden SQL을 의도에 맞게 조정"""
        sql = base_sql
        
        # 시간 범위 조정
        if intent.get("time_range"):
            # WHERE 절 수정 또는 추가
            if "WHERE" in sql.upper():
                sql = sql.replace("WHERE", f"WHERE production_date >= {intent['time_range']} AND")
            else:
                sql += f"\nWHERE production_date >= {intent['time_range']}"
        
        # 필터 조정
        filter_data = intent.get("filter")
        if filter_data:
            field = filter_data.get("field", "").lower()
            value = filter_data.get("value")
            if "WHERE" in sql.upper():
                sql += f"\n  AND {field} = '{value}'"
            else:
                sql += f"\nWHERE {field} = '{value}'"
        
        return sql
    
    def _adapt_kpi_sql(self, base_sql: str, intent: Dict) -> str:
        """KPI SQL을 의도에 맞게 조정"""
        return self._adapt_golden_sql(base_sql, intent)
    
    def _generate_basic_sql(self, intent: Dict, tables: List[Dict], normalized_data: Dict) -> str:
        """기본 SQL 쿼리 생성"""
        
        # 메트릭에 따른 테이블 선택
        metric = intent.get("metric")
        table_mappings = {
            "DEFECT_RATE": "quality_inspection",
            "YIELD_RATE": "production_results",
            "OEE": "equipment_performance",
            "PRODUCTION_QUANTITY": "production_results",
            "INVENTORY": "inventory",
            "MTBF": "equipment_failures",
            "MTTR": "equipment_failures",
        }
        
        table_name = table_mappings.get(metric)
        if not table_name and tables:
            table_name = tables[0]['table_name']
        elif not table_name:
            table_name = "production_results"
        
        # SELECT 절 구성
        select_cols = []
        grouping = intent.get("grouping")
        
        if grouping:
            select_cols.append(grouping)
        
        # 메트릭 계산
        aggregation = intent.get("aggregation", "AVG")
        if metric == "DEFECT_RATE":
            select_cols.append(f"{aggregation}(defect_quantity / total_quantity * 100) AS defect_rate")
        elif metric == "OEE":
            select_cols.append(f"{aggregation}(oee) AS avg_oee")
        elif metric == "PRODUCTION_QUANTITY":
            select_cols.append(f"{aggregation}(production_quantity) AS total_production")
        else:
            select_cols.append(f"{aggregation}(*) AS value")
        
        select_clause = ", ".join(select_cols) if select_cols else "*"
        
        # 기본 쿼리 구성
        sql = f"SELECT {select_clause}\nFROM {table_name}"
        
        # WHERE 절
        where_conditions = []
        
        # 시간 범위
        if intent.get("time_range"):
            where_conditions.append(f"production_date >= {intent['time_range']}")
        
        # 필터
        filter_data = intent.get("filter")
        if filter_data:
            field = filter_data.get("field", "").lower()
            value = filter_data.get("value")
            where_conditions.append(f"{field} = '{value}'")
        
        if where_conditions:
            sql += "\nWHERE " + " AND ".join(where_conditions)
        
        # GROUP BY 절
        if grouping and aggregation:
            sql += f"\nGROUP BY {grouping}"
        
        # ORDER BY 절
        if intent.get("ordering"):
            order_col = "value" if not metric else select_cols[-1].split(" AS ")[-1]
            sql += f"\nORDER BY {order_col} {intent['ordering']}"
        
        # LIMIT 절
        if intent.get("limit"):
            sql += f"\nLIMIT {intent['limit']}"
        
        return sql


def process_json_file(json_file_path: str, rag_system: ManufacturingRAGSystem) -> Dict:
    """JSON 파일을 읽어서 SQL 생성"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        generator = SQLQueryGenerator(rag_system)
        result = generator.generate_sql_from_json(json_data)
        
        # 메타데이터 추가
        if "metadata" in json_data:
            result["metadata"] = json_data["metadata"]
        
        return result
    
    except Exception as e:
        return {
            "error": str(e),
            "json_file": json_file_path
        }


def main():
    """메인 함수"""
    print("=" * 80)
    print("🏭 제조업 RAG 기반 SQL 쿼리 생성기 (JSON 입력 버전)")
    print("=" * 80)
    
    # RAG 시스템 초기화
    print("\n[1] RAG 시스템 초기화 중...")
    rag_system = ManufacturingRAGSystem()
    
    # 데이터 로드 (경로 확인 필요)
    try:
        rag_system.load_schema_data("./input/RAG/schema_Info.json")
        rag_system.load_kpi_data("./input/RAG/kpi.json")
        rag_system.load_golden_sql_data("./input/RAG/goldenSQL.json")
        print("   ✅ RAG 데이터 로드 완료")
    except Exception as e:
        print(f"   ⚠️ RAG 데이터 로드 실패: {e}")
        print("   ℹ️ 기본 쿼리 생성 모드로 동작합니다.")
    
    # output 폴더에서 JSON 파일 찾기
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"\n❌ 오류: '{output_dir}' 폴더가 존재하지 않습니다.")
        return
    
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"\n❌ 오류: '{output_dir}' 폴더에 JSON 파일이 없습니다.")
        return
    
    # 가장 최근 파일 선택
    json_files.sort(reverse=True)
    latest_file = os.path.join(output_dir, json_files[0])
    
    print(f"\n[2] JSON 파일 처리 중...")
    print(f"   📁 파일: {latest_file}")
    
    # SQL 생성
    result = process_json_file(latest_file, rag_system)
    
    if "error" in result:
        print(f"\n❌ 오류 발생: {result['error']}")
        return
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("📝 원본 질의")
    print("=" * 80)
    print(result.get("original_query", "N/A"))
    
    print("\n" + "=" * 80)
    print("📊 정규화된 데이터")
    print("=" * 80)
    print(json.dumps(result["normalized_data"], indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80)
    print("🔍 의도 분석")
    print("=" * 80)
    intent = result["intent_analysis"]
    print(f"   의도 유형: {intent.get('intent_type')}")
    print(f"   메트릭: {intent.get('metric')}")
    print(f"   시간범위: {intent.get('time_range')}")
    print(f"   필터: {intent.get('filter')}")
    print(f"   집계함수: {intent.get('aggregation')}")
    print(f"   그룹핑: {intent.get('grouping')}")
    print(f"   정렬: {intent.get('ordering')}")
    print(f"   LIMIT: {intent.get('limit')}")
    
    print("\n" + "=" * 80)
    print("🔎 RAG 검색 결과")
    print("=" * 80)
    print(f"   관련 테이블: {', '.join(result['relevant_tables'][:5]) if result['relevant_tables'] else 'N/A'}")
    print(f"   관련 KPI: {', '.join(result['relevant_kpis'][:5]) if result['relevant_kpis'] else 'N/A'}")
    
    print("\n" + "=" * 80)
    print(f"💡 추천 SQL (출처: {result['sql_source']}, 신뢰도: {result['confidence']:.0f}%)")
    print("=" * 80)
    if result['recommended_sql']:
        sql_lines = result['recommended_sql'].split('\n')
        for line in sql_lines:
            print(f"   {line}")
    else:
        print("   SQL 생성 실패")
    
    print("\n" + "=" * 80)
    print("📌 설명")
    print("=" * 80)
    print(f"   {result['explanation']}")
    
    # 메타데이터 출력
    if result.get("metadata"):
        print("\n" + "=" * 80)
        print("ℹ️ 메타데이터")
        print("=" * 80)
        metadata = result["metadata"]
        print(f"   타임스탬프: {metadata.get('timestamp', 'N/A')}")
        print(f"   모델: {metadata.get('model', 'N/A')}")
        print(f"   총 쿼리 수: {metadata.get('total_queries', 'N/A')}")
        print(f"   용어집 크기: {metadata.get('glossary_terms', 'N/A')}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()