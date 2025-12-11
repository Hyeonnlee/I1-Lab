import os
import json
import pandas as pd
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def read_csv_files(folder_path):
    """input/sample_query_data 폴더의 CSV 파일들을 읽습니다."""
    csv_data = []
    
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            csv_data.append({
                'filename': file,
                'filepath': file_path,
                'dataframe': df,
                'preview': df.head(10).to_string(),
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'shape': df.shape
            })
    
    return csv_data

def read_domain_glossary(json_path):
    """domain_glossary.json 파일을 읽습니다."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_visualization_code(csv_data, glossary, client, model_id):
    """Hugging Face Inference Client를 사용하여 시각화 코드를 생성합니다."""
    
    # CSV 데이터 정보 준비
    data_info = []
    for data in csv_data:
        info = f"""
파일명: {data['filename']}
컬럼: {', '.join(data['columns'])}
데이터 타입: {data['dtypes']}
행/열 개수: {data['shape']}
데이터 미리보기:
{data['preview']}
"""
        data_info.append(info)
    
    # 파일 경로 정보 추가
    file_paths = {}
    for data in csv_data:
        file_paths[data['filename']] = data['filepath']
    
    system_message = """You are a Python data visualization expert.
Generate matplotlib and seaborn visualization code based on CSV data.

CRITICAL RULES:
1. Use FULL file paths when reading CSV files (pd.read_csv())
2. Analyze CSV columns and data types to select appropriate visualizations
3. Use matplotlib and seaborn
4. Include Korean font settings: plt.rcParams['font.family'] = 'Malgun Gothic'
5. Add clear titles and axis labels
6. Save plots to 'output/' folder with descriptive names
7. Return ONLY executable Python code without explanations
8. Start with all necessary imports including: import os"""

    user_message = f"""CSV 파일 경로:
{json.dumps(file_paths, ensure_ascii=False, indent=2)}

CSV 데이터 정보:
{chr(10).join(data_info)}

Domain Glossary:
{json.dumps(glossary, ensure_ascii=False, indent=2)}

위 데이터를 분석하여 적절한 matplotlib과 seaborn 시각화 코드를 생성하세요.

IMPORTANT:
- CSV 파일을 읽을 때 반드시 위의 '파일 경로' 정보의 FULL PATH를 사용하세요
- 예: pd.read_csv('input/sample_query_data/production_orders.csv')
- PNG 파일은 'output/' 폴더에 저장하세요
- os.makedirs('output', exist_ok=True) 코드를 포함하세요
- 한글 폰트 설정 포함
- 실행 가능한 완전한 Python 코드만 출력하세요"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    try:
        print("   Chat Completion API 호출 중...")
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=2000,
            temperature=0.3
        )
        response_text = response.choices[0].message.content
        print("   ✓ API 응답 받음")
        
    except Exception as e:
        print(f"   ✗ Chat API 실패: {e}")
        print("   🔄 Text Generation으로 재시도...")
        
        prompt = f"""{system_message}

{user_message}

Python Code:"""
        
        response = client.text_generation(
            prompt,
            model=model_id,
            max_new_tokens=2000,
            temperature=0.3,
            return_full_text=False
        )
        response_text = response
        print("   ✓ 재시도 성공")
    
    # 코드 추출
    code = clean_code_output(response_text)
    return code

def clean_code_output(text):
    """LLM 출력에서 Python 코드 추출"""
    # 마크다운 코드 블록 제거
    text = re.sub(r'```python\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # import 문부터 시작하는 코드 찾기
    if 'import' in text:
        start = text.find('import')
        if start != -1:
            return text[start:].strip()
    
    return text.strip()

def save_visualization_code(code, output_folder, filename='visualization.py'):
    """생성된 코드를 output 폴더에 저장합니다."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return output_path

def main():
    # 경로 설정
    csv_folder = 'input/sample_querydata'
    glossary_path = 'input/domain_glossary.json'
    output_folder = 'output'
    
    # API 토큰 확인
    api_token = os.getenv("HUGGINGFACE_API_KEY")
    if not api_token:
        print("❌ HUGGINGFACE_API_KEY가 설정되지 않았습니다.")
        print("다음 명령어로 설정하세요:")
        print("  export HUGGINGFACE_API_KEY='your-token-here'")
        return
    
    print("=" * 60)
    print("데이터 시각화 코드 생성 에이전트 (Hugging Face)")
    print("=" * 60)
    
    # Inference Client 초기화
    print(f"\n[0단계] Hugging Face Client 초기화")
    try:
        client = InferenceClient(token=api_token)
        print("   ✓ 클라이언트 초기화 성공")
    except Exception as e:
        print(f"   ✗ 초기화 실패: {e}")
        return
    
    # 사용할 모델 목록 (우선순위 순)
    model_options = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/flan-t5-large",
        "bigcode/starcoder2-3b",
    ]
    
    selected_model = model_options[0]
    print(f"   ✓ 사용 모델: {selected_model}")
    
    # CSV 파일 읽기
    print(f"\n[1단계] CSV 파일 읽기: {csv_folder}")
    try:
        csv_data = read_csv_files(csv_folder)
        print(f"   ✓ {len(csv_data)}개의 CSV 파일을 찾았습니다.")
        for data in csv_data:
            print(f"     - {data['filename']} ({data['shape'][0]} rows, {data['shape'][1]} cols)")
    except Exception as e:
        print(f"   ✗ CSV 읽기 실패: {e}")
        return
    
    # Domain Glossary 읽기
    print(f"\n[2단계] Domain Glossary 읽기: {glossary_path}")
    try:
        glossary = read_domain_glossary(glossary_path)
        print(f"   ✓ Domain Glossary 로드 완료")
    except Exception as e:
        print(f"   ✗ Glossary 읽기 실패: {e}")
        return
    
    # AI 에이전트로 시각화 코드 생성
    print(f"\n[3단계] AI 에이전트를 통한 시각화 코드 생성 중...")
    try:
        visualization_code = generate_visualization_code(
            csv_data, 
            glossary, 
            client, 
            selected_model
        )
        print(f"   ✓ 시각화 코드 생성 완료 ({len(visualization_code)} 문자)")
    except Exception as e:
        print(f"   ✗ 코드 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 코드 저장
    print(f"\n[4단계] 생성된 코드 저장: {output_folder}")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_{timestamp}.py"
        output_path = save_visualization_code(visualization_code, output_folder, filename)
        print(f"   ✓ 저장 완료: {output_path}")
    except Exception as e:
        print(f"   ✗ 저장 실패: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ 작업 완료!")
    print("=" * 60)
    print(f"\n생성된 파일을 실행하려면:")
    print(f"  python {output_path}")
    print()

if __name__ == "__main__":
    main()