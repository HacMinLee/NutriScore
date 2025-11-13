"""
Project Swan's Eye v4.5 - Main Dashboard (Control Panel + A/B Testing Studio)
- v4.9.3 (사장님 요청): [Tab 1] '최종 순위' 표시에 '누락'된 '브랜드'/'MarketScore'를 '표시'하고,
    - '컬럼 순서'를 ('브랜드', '제품명', '영양제점수', '가격', 'MarketScore', '그외')로 '재배치'.
    - `initialize_session_state`에 '브랜드' 컬럼('brand': '브랜드') '추가'.
- v4.8.1 (치명적 버그 수정): 'apply_filters'의 "배제" -> "배제" 'Blackbox' 버그 수정.
- v4.8 (사장님 요청): '부수적인 버튼' 'st.container(border=True)' 적용.
- v4.7 (사장님 요청): 필터 UI의 "쓸데없는말" (예: "~ 성분 필터") '싹 다 삭제'.
- v4.6 (사장님 요청): "b1이 있는제품만" 필터링 되도록 '성분 필터' 로직 '완벽' 수정.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import core_engine_v2 as core_engine # v2.7.1 (v4.9.3) 엔진 임포트
import copy
import plotly.express as px

# ---
# 페이지 기본 설정
# ---
st.set_page_config(
    page_title="영양제의정석",
    layout="wide"
)

# ---
# [v4.5] CSV 자동 스캐너 (v2.6 확장판)
# ---
@st.cache_data # CSV 스캔은 한번만
def scan_csv_for_rules_v4_5(df):
    """
    CSV를 스캔하여 '핵심/보조/태그' 뿐만 아니라,
    '브랜드' 등 '텍스트(Object)' 컬럼의 고유값도 '싹 다' 스캔.
    """
    
    # [v2.6.1] None 방어 코드
    if df is None:
        st.warning("scan_csv_for_rules: CSV 데이터가 없어 스캔을 건너뜁니다.")
        return {'main_comps': [], 'sub_comps': [], 'tags': [], 'text_cols': {}}
    
    discovered = {
        'main_comps': set(),
        'sub_comps': set(),
        'tags': set(),
        'text_cols': {} # [v4.5 신규] '브랜드' 등을 담을 곳
    }
    
    # 1. 성분 스캔 (핵심, 보조)
    comp_cols = ['핵심성분명태그', '보조성분명태그']
    pattern = re.compile(r"성분\s*:\s*([^,]+)", re.IGNORECASE)
    
    for col in comp_cols:
        if col in df.columns:
            for text in df[col].dropna():
                clean_text = str(text).replace(" ", "")
                match = pattern.search(clean_text)
                if match:
                    comp_name = match.group(1).strip()
                    if comp_name:
                        if col == '핵심성분명태그':
                            discovered['main_comps'].add(comp_name)
                        else:
                            discovered['sub_comps'].add(comp_name)

    # 2. 태그 스캔
    tag_col = '특수태그'
    if tag_col in df.columns:
        # [v2.7.2] 버그 수정된 로직
        for text in df[tag_col].dropna(): 
            tags_list = str(text).split('|')
            for tag in tags_list:
                clean_tag = tag.strip().replace('*', '').strip()
                if clean_tag:
                    discovered['tags'].add(clean_tag)

    # 3. [v4.5 신규] '브랜드' 등 텍스트 컬럼 스캔
    # (핵심 로직에서 이미 사용 중인 컬럼은 제외)
    excluded_cols = [
        '제품명', '핵심성분명태그', '보조성분명태그', '특수태그',
        '1일 섭취량당 가격', '리뷰 개수', '리뷰 별점'
    ]
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col not in excluded_cols:
            unique_values = df[col].dropna().unique()
            # [v4.9.3] '브랜드' 컬럼이 50개 이상이어도 스캔되도록 50->100으로 확장
            if 1 < len(unique_values) < 100: 
                discovered['text_cols'][col] = sorted(list(unique_values))
                    
    return {
        'main_comps': sorted(list(discovered['main_comps'])),
        'sub_comps': sorted(list(discovered['sub_comps'])),
        'tags': sorted(list(discovered['tags'])),
        'text_cols': discovered['text_cols'] # 딕셔너리 { '브랜드': ['A', 'B'], ... }
    }

# ---
# [v4.9.3] 세션 상태 초기화 ('브랜드' '누락' 복구)
# ---
def initialize_session_state(discovered_rules):
    """
    '자동 발견된 목록'으로 v2.7 룰북의 기본 구조를 생성합니다.
    """
    if 'v2_rulebook' in st.session_state:
        return # 이미 초기화됨

    rb = {
        'columns': { # v1.4의 공통 컬럼
            'product_name': '제품명',
            'price': '1일 섭취량당 가격',
            'review_count': '리뷰 개수',
            'rating': '리뷰 별점',
            'brand': '브랜드' # --- [v4.9.3] '브랜드' '누락' 복구 ---
        },
        'final_weights': { 'weight_a': 0.5, 'weight_b': 0.3, 'weight_c': 0.2 },
        'score_a_main_components': {
            'csv_column': '핵심성분명태그',
            'rules': {}
        },
        'score_b_price': { 'k_value': 1.0 },
        'score_c_sub_components': {
            'csv_column': '보조성분명태그',
            'final_weight': 0.5,
            'rules': {}
        },
        'score_c_tags': {
            'csv_column': '특수태그',
            'final_weight': 0.5,
            'rules': {}
        },
        # [v2.7]  분석기용 룰
        'market_score_weights': {
            'k_review': 2.0, # v1.4 기본값
            'k_rating': 1.0, # v1.4 기본값
            'weight_review': 0.7, # v1.4 기본값
            'weight_rating': 0.3  # v1.4 기본값
        }
    }

    # 1. Score A 룰북 채우기 (v2.6.4: 'enabled': True)
    for name in discovered_rules['main_comps']:
        rb['score_a_main_components']['rules'][name] = {
            'enabled': True,
            'min_dose': 500.0, 'rec_dose': 1000.0,
            'rec_score': 80.0, 'saturation_factor': 1.0,
            'weight': 1.0
        }
        
    # 2. Score C-1 룰북 채우기 (v2.6.4: 'enabled': True)
    for name in discovered_rules['sub_comps']:
        rb['score_c_sub_components']['rules'][name] = {
            'enabled': True,
            'min_dose': 100.0, 'rec_dose': 200.0,
            'rec_score': 70.0, 'saturation_factor': 0.5,
            'weight': 1.0
        }

    # 3. Score C-2 룰북 채우기 (점수 0)
    for name in discovered_rules['tags']:
        rb['score_c_tags']['rules'][name] = 0.0
        
    st.session_state.v2_rulebook = rb
    
    # [v4.5 신규] '필터' UI의 상태를 저장할 공간 (룰북과 분리)
    if 'v4_filters_A' not in st.session_state:
        st.session_state.v4_filters_A = {}
    if 'v4_filters_B' not in st.session_state:
        st.session_state.v4_filters_B = {}

# ---
# [v4.8] 헬퍼 함수 1: '다중 필터 박스' UI (v4.7 '쓸데없는말' 제거 + v4.8 '컨테이너' 적용)
# ---
def create_filter_box(box_id, discovered_rules, delta_df):
    """
    [v4.8] '하나로 통일'된 v4.5 '다중 필터 박스' UI를 생성합니다.
    "필터걸지말지" (Checkbox) + "추가 조정" (Radio/Slider/Multiselect) 로직 구현.
    '부수적인 버튼'을 'st.container(border=True)'로 "가시적"으로 "구분".
    """
    
    # 필터 상태 저장을 위해 세션 상태 사용
    if box_id not in st.session_state:
        st.session_state[box_id] = {}

    filter_state = st.session_state[box_id]

    # 1. 성분(핵심/보조) 필터 (Slider) --- [v4.6] "필터걸지말지" + "추가조정" 로직으로 수정
    with st.expander("🔬 1. 성분 함량(스펙) 필터"):
        all_components = discovered_rules['main_comps'] + discovered_rules['sub_comps']
        for comp_name in all_components:
            if comp_name not in delta_df.columns:
                continue
            
            # (1) "필터걸지말지" Checkbox (v4.7 - "쓸데없는말" 제거)
            use_filter = st.checkbox(f"'{comp_name}'", key=f"check_{box_id}_{comp_name}")
            
            filter_rule = {} # (v4.6) 필터 룰을 딕셔너리로 저장
            
            if use_filter:
                # --- [v4.8] "부수적인버튼"을 "가시적"으로 "구분"하기 위해 '컨테이너' 추가 ---
                with st.container(border=True):
                    # (2) "b1이 있는제품만" (포함/미포함) Radio
                    rule_type = st.radio(
                        f"'{comp_name}' 포함 여부:",
                        ["반드시 포함", "배제"],
                        key=f"radio_{box_id}_{comp_name}",
                        horizontal=True
                    )
                    filter_rule['type'] = rule_type
                    
                    # (3) "추가로 조정할려면 하고" Checkbox
                    use_slider = st.checkbox("함량 범위 필터", key=f"check_slider_{box_id}_{comp_name}")
                    
                    if use_slider:
                        # (4) "추가로 조정" Slider
                        min_val, max_val = float(delta_df[comp_name].min()), float(delta_df[comp_name].max())
                        if pd.isna(min_val) or pd.isna(max_val):
                            st.caption(f"'{comp_name}' 데이터가 없어 함량 범위를 조정할 수 없습니다.")
                            filter_rule['slider'] = None
                        else:
                            # [v4.3.1] min/max 동일 값 오류 수정
                            min_val = float(min_val)
                            max_val = float(max_val)
                            if max_val <= min_val: max_val = min_val + 1.0 
                            
                            filter_rule['slider'] = st.slider(
                                f"'{comp_name}' 함량 범위:",
                                min_value=min_val, max_value=max_val,
                                value=(min_val, max_val),
                                key=f"slider_{box_id}_{comp_name}"
                            )
                    else:
                        filter_rule['slider'] = None # "추가 조정" 안 함
                # --- [v4.8] 컨테이너 끝 ---
                
                filter_state[comp_name] = filter_rule # { 'type': '반드시 포함', 'slider': (800, 1200) }
                
            elif comp_name in filter_state:
                del filter_state[comp_name] # 체크 해제 시 필터 룰 삭제

    # 2. 특수태그 필터 (Radio)
    with st.expander("🔬 2. 특수태그 필터"):
        for tag_name in discovered_rules['tags']:
            # '필터 걸지 말지' Checkbox (v4.7 - "쓸데없는말" 제거)
            use_filter = st.checkbox(f"'{tag_name}'", key=f"check_{box_id}_{tag_name}")
            
            if use_filter:
                # --- [v4.8] "부수적인버튼" '컨테이너' 추가 ---
                with st.container(border=True):
                    # '추가로 조정' Radio
                    filter_state[tag_name] = st.radio(
                        f"'{tag_name}' 포함 여부:",
                        ["반드시 포함", "배제"],
                        key=f"radio_{box_id}_{tag_name}",
                        horizontal=True
                    )
                # --- [v4.8] 컨테이너 끝 ---
            elif tag_name in filter_state:
                del filter_state[tag_name] # 체크 해제 시 필터 룰 삭제

    # 3. 텍스트('브랜드' 등) 필터 (Multiselect)
    with st.expander("🔬 3. '브랜드' 등 텍스트 필터"):
        for col_name, options in discovered_rules['text_cols'].items():
            # '필터 걸지 말지' Checkbox (v4.7 - "쓸데없는말" 제거)
            use_filter = st.checkbox(f"'{col_name}'", key=f"check_{box_id}_{col_name}")
            
            if use_filter:
                # --- [v4.8] "부수적인버튼" '컨테이너' 추가 ---
                with st.container(border=True):
                    # '추가로 조정' Multiselect
                    filter_state[col_name] = st.multiselect(
                        f"'{col_name}'에서 포함할 항목:",
                        options=options,
                        default=options, # 기본값 = 전체 선택
                        key=f"multi_{box_id}_{col_name}"
                    )
                # --- [v4.8] 컨테이너 끝 ---
            elif col_name in filter_state:
                del filter_state[col_name] # 체크 해제 시 필터 룰 삭제

    return filter_state # { 'EPA+DHA': {'type':'반드시 포함', 'slider':(800,1200)}, 'rtg여부': '반드시 포함', '브랜드': ['A', 'B'] }

# ---
# [v4.8.1] 헬퍼 함수 2: '다중 필터' 적용 (v4.6.1 'Blackbox' 버그 수정)
# ---
def apply_filters(df, filters):
    """
    [v4.8.1] '다중 필터' 룰(v4.6 성분 룰)을 받아 '엑셀' '노가다'를 '자동화'합니다.
    (v4.8.1) "배제" -> "배제" 'Blackbox' 버그 '완벽' 수정.
    """
    filtered_df = df.copy()
    
    for key, rule in filters.items():
        
        # --- [v4.6] 성분 룰 (dict) 처리 ---
        if isinstance(rule, dict):
            # (1) "b1이 있는제품만" (포함/미포함) 필터
            if rule['type'] == "반드시 포함":
                filtered_df = filtered_df[filtered_df[key].notna()]
            elif rule['type'] == "배제": # <-- [v4.8.1] "배제"에서 수정
                filtered_df = filtered_df[filtered_df[key].isna()]
            
            # (2) "추가로 조정" Slider 필터 (선택 사항)
            if rule['slider'] is not None:
                min_val, max_val = rule['slider']
                # (주의: 'notna'/'isna'로 이미 걸러졌으므로, 'isna()' OR 조건 제거)
                filtered_df = filtered_df[
                    (filtered_df[key].between(min_val, max_val))
                ]
        # --- [수정 완료] ---
                
        elif rule == "반드시 포함": # 특수태그 (v4.5와 동일)
            mask = filtered_df['tags_raw'].str.contains(
                f"{re.escape(key)}\s*\*", na=False, regex=True
            )
            filtered_df = filtered_df[mask]
        elif rule == "배제": # 특수태그 <-- [v4.8.1] "배제"에서 수정
            mask = filtered_df['tags_raw'].str.contains(
                f"{re.escape(key)}\s*\*", na=False, regex=True
            )
            filtered_df = filtered_df[~mask]
        elif isinstance(rule, list): # ['A', 'B'] -> 텍스트/브랜드 Multiselect (v4.5와 동일)
            mask = filtered_df[key].isin(rule)
            filtered_df = filtered_df[mask]
            
    return filtered_df

# ---
# [메인 프로그램]
# ---
st.title("영양제의정석")

# ---
# [0] CSV 파일 업로드
# ---
uploaded_file = st.file_uploader("CSV 파일 선택 ('제품관리 엑셀추출.csv' 양식)", type=["csv"])
if uploaded_file is None:
    st.info("⬆️ 분석할 CSV 파일을 업로드해 주세요.")
    st.stop()

# [v2.6.2] 수정된 로더
@st.cache_data
def load_csv(file):
    try:
        # 1차: UTF-8 우선 시도
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # 2차: cp949 시도
            file.seek(0) # 파일 포인터 리셋
            return pd.read_csv(file, encoding='cp949') # 실패 시 cp949
        except Exception as e:
            # 2차 시도(cp949)도 실패
            st.error(f"파일 로드 오류 (cp949 시도): {e}")
            return None
    except Exception as e:
        # 1차 시도(utf-8)에서 UnicodeDecodeError 외의 오류 발생
        st.error(f"파일 로드 오류 (utf-8 시도): {e}")
        return None

raw_df = load_csv(uploaded_file)
if raw_df is None:
    st.error("CSV 파일 로드에 최종 실패했습니다. 파일 인코딩(utf-8, cp949)이나 내용을 확인해 주세요.")
    st.stop() 

# [v4.5] 스캐너 실행 (v4.5) 및 세션 초기화 (v4.9.3)
try:
    # _discovered_rules는 @st.cache_data로 캐시됨
    _discovered_rules = scan_csv_for_rules_v4_5(raw_df)
    initialize_session_state(_discovered_rules)
except KeyError as e:
    st.error(f"CSV 스캔 오류: '{e}' 컬럼이 없습니다.")
    st.stop()
    
# 편의를 위해 세션 룰북 변수 할당
rb = st.session_state.v2_rulebook

# ---
# [v2.7] 탭(Tabs) UI
# ---
tab1, tab2 = st.tabs(["🕹️ 컨트롤 패널", "🔬 'A/B 테스팅'"])


# ---
# [TAB 1] 컨트롤 패널 ("그대로 냅두고")
# ---
with tab1:
    st.header("🕹️ 컨트롤 패널")
    st.write("점수 함수 설정.")
    
    # --- [1] 최종 점수 가중치 (Final Weights) --- [v2.6.4]
    st.subheader("1. 최종 점수 가중치 (A+B+C)")
    fw = rb['final_weights']
    fw_cols = st.columns(3)
    fw['weight_a'] = fw_cols[0].number_input("A: 핵심성분 비중", 0.0, value=fw['weight_a'], step=0.1)
    fw['weight_b'] = fw_cols[1].number_input("B: 가격 비중", 0.0, value=fw['weight_b'], step=0.1)
    fw['weight_c'] = fw_cols[2].number_input("C: 보조/태그 비중", 0.0, value=fw['weight_c'], step=0.1)
    total_w = fw['weight_a'] + fw['weight_b'] + fw['weight_c']
    if total_w == 0: total_w = 1.0
    st.info(f"적용 비율 (자동 정규화) | 핵심성분: **{fw['weight_a']/total_w*100 :.1f}%** | 가 격: **{fw['weight_b']/total_w*100 :.1f}%** | 보조/태그: **{fw['weight_c']/total_w*100 :.1f}%**")

    st.divider()

    # --- [2] Score A: 핵심성분 편집기 --- [v2.6.4]
    st.subheader(f"2. [Score A] 핵심성분 편집기 (from: '{rb['score_a_main_components']['csv_column']}')")
    sa_rules = rb['score_a_main_components']['rules']
    if not sa_rules:
        st.warning(f"'{rb['score_a_main_components']['csv_column']}' 컬럼에서 '성분:이름'을 찾지 못했습니다.")
    else:
        for comp_name, rule in sa_rules.items():
            with st.expander(f"**{comp_name}**", expanded=rule['enabled']):
                rule['enabled'] = st.toggle("✅ 이 성분으로 점수 계산", value=rule['enabled'], key=f"a_en_{comp_name}")
                sc_cols = st.columns(4)
                rule['min_dose'] = sc_cols[0].number_input("최저(min)", value=rule['min_dose'], key=f"a_min_{comp_name}")
                rule['rec_dose'] = sc_cols[1].number_input("권장(rec)", value=rule['rec_dose'], key=f"a_rec_{comp_name}")
                rule['rec_score'] = sc_cols[2].number_input("권장점수(score)", value=rule['rec_score'], key=f"a_sc_{comp_name}")
                rule['saturation_factor'] = sc_cols[3].number_input("포화계수(k)", value=rule['saturation_factor'], format="%.2f", key=f"a_sat_{comp_name}")
                rule['weight'] = st.slider("내부 가중치", 0.0, 1.0, value=rule['weight'], key=f"a_w_{comp_name}")

    st.divider()

    # --- [3] Score B: 가격 편집기 (공통) ---
    st.subheader("3. [Score B] 가격 편집기 (Z-Score)")
    rb['score_b_price']['k_value'] = st.number_input(
        "S-Curve 기울기(k)", value=rb['score_b_price']['k_value'], help="높을수록 고가/저가 차이가 큼"
    )
    st.caption(f"적용 컬럼: '{rb['columns']['price']}'")

    st.divider()

    # --- [4] Score C: 보조성분 / 태그 편집기 ---
    st.subheader("4. [Score C] 보조성분 및 태그 편집")
    st.write("**Score C 내부 가중치**")
    sc_w_cols = st.columns(2)
    rb['score_c_sub_components']['final_weight'] = sc_w_cols[0].number_input(
      "C-1 (보조성분 S-Curve) 비중", 0.0, value=rb['score_c_sub_components']['final_weight'], step=0.1
    )
    rb['score_c_tags']['final_weight'] = sc_w_cols[1].number_input(
        "C-2 (특수태그 합산) 비중", 0.0, value=rb['score_c_tags']['final_weight'], step=0.1
    )

    # --- (C-1: 보조성분 S-Curve) --- [v2.6.4]
    st.markdown("---")
    st.write(f"**C-1: 보조성분 (from: '{rb['score_c_sub_components']['csv_column']}')**")
    sc1_rules = rb['score_c_sub_components']['rules']
    if not sc1_rules:
        st.warning(f"'{rb['score_c_sub_components']['csv_column']}' 컬럼에서 '성분:이름'을 찾지 못했습니다.")
    else:
        for comp_name, rule in sc1_rules.items():
            with st.expander(f"**{comp_name}**", expanded=rule['enabled']):
                rule['enabled'] = st.toggle("✅ 이 성분으로 점수 계산", value=rule['enabled'], key=f"c1_en_{comp_name}")
                sc_cols = st.columns(4)
                rule['min_dose'] = sc_cols[0].number_input("최저(min)", value=rule['min_dose'], key=f"c1_min_{comp_name}")
                rule['rec_dose'] = sc_cols[1].number_input("권장(rec)", value=rule['rec_dose'], key=f"c1_rec_{comp_name}")
                rule['rec_score'] = sc_cols[2].number_input("권장점수(score)", value=rule['rec_score'], key=f"c1_sc_{comp_name}")
                rule['saturation_factor'] = sc_cols[3].number_input("포화계수(k)", value=rule['saturation_factor'], format="%.2f", key=f"c1_sat_{comp_name}")
                rule['weight'] = st.slider("내부 가중치", 0.0, 1.0, value=rule['weight'], key=f"c1_w_{comp_name}")

    # --- (C-2: 특수태그 합산) ---
    st.markdown("---")
    st.write(f"**C-2: 특수태그 (합산 점수) (from: '{rb['score_c_tags']['csv_column']}')**")
    st.caption("점수가 0이면 무시됩니다. '*'가 태그명 뒤에 붙어야 적용됩니다.")
    sc2_rules = rb['score_c_tags']['rules']
    if not sc2_rules:
        st.warning(f"'{rb['score_c_tags']['csv_column']}' 컬럼에서 태그를 찾지 못했습니다.")
    else:
        tag_cols = st.columns(3)
        col_idx = 0
        for tag_name, tag_score in sc2_rules.items():
            with tag_cols[col_idx % 3]:
                sc2_rules[tag_name] = st.number_input(
                    f"'{tag_name}' 점수", value=float(tag_score), step=0.5, key=f"c2_tag_{tag_name}"
                )
            col_idx += 1
            
    st.divider()

    # --- [v2.7]  분석기용 Market Score 튜닝 ---
    st.subheader("5. Market Score 튜닝")
    msw = rb['market_score_weights']
    msw_cols = st.columns(4)
    msw['k_review'] = msw_cols[0].number_input("리뷰 수 k (기울기)", 0.1, value=msw['k_review'], step=0.1)
    msw['weight_review'] = msw_cols[1].number_input("리뷰 수 비중", 0.0, value=msw['weight_review'], step=0.1)
    msw['k_rating'] = msw_cols[2].number_input("별점 k (기울기)", 0.1, value=msw['k_rating'], step=0.1)
    msw['weight_rating'] = msw_cols[3].number_input("별점 비중", 0.0, value=msw['weight_rating'], step=0.1)
    st.caption(f"적용 컬럼: '{rb['columns']['review_count']}', '{rb['columns']['rating']}'")

    st.divider()

    # --- [v4.9.3] 6. 분석 실행 (컬럼 순서 재배치) ---
    st.header("📈 분석결과")
    if st.button("▶️ 분석 실행하기", type="primary"):
        dynamic_rulebook = copy.deepcopy(st.session_state.v2_rulebook)
        st.write("---")
        st.subheader("적용된 최종 룰북 (JSON)")
        st.json(dynamic_rulebook, expanded=False)
        try:
            with st.spinner(""):
                final_df = core_engine.run_full_analysis_v2_6(raw_df, dynamic_rulebook)
            st.subheader("최종 순위 및 점수")
            
            # --- [v4.9.3 수정] ---
            # (1) 'Blackbox' 없는 '이름 변경' (v4.9.2 확장)
            final_df = final_df.rename(columns={
                'SWAN_SCORE_V2': '영양제점수',
                'product_name': '제품명',
                'price': '가격',
                'MARKET_SCORE': 'MarketScore'
                # '브랜드'는 엔진(v4.9.3)에서 '브랜드'로 '추가'됨
            })
            
            # (2) 사장님이 요청하신 "원하는 순서" ('그 외' 포함)
            desired_order = [
                '브랜드', 
                '제품명', 
                '영양제점수', 
                '가격', 
                'MarketScore'
            ]
            
            # (3) '그 외' 컬럼 '자동' 추가 (순서 유지)
            existing_cols = [col for col in desired_order if col in final_df.columns]
            other_cols = [col for col in final_df.columns if col not in existing_cols]
            final_display_cols = existing_cols + other_cols
            # --- [v4.9.3 수정 완료] ---

            st.dataframe(final_df[final_display_cols].style.format(precision=2))
            
        except ValueError as e:
            st.error(f"엔진 실행 중 오류가 발생했습니다: {e}")
        except Exception as e:
            st.error(f"알 수 없는 심각한 오류: {e}")


# ---
# [TAB 2] v4.8 'A/B 테스팅' 델타 분석기 (v4.8.1 버그 수정)
# ---
with tab2:
    st.header("🔬 A/B 테스팅")
    st.write("""
    "다중필터
    """)

    # --- [v2.7] 델타 분석기용 데이터 준비 ---
    @st.cache_data
    def prepare_delta_data(_raw_df, _rules):
        """
        전처리(agg_df) 및 Market Score 계산을 수행하여 델타 분석용 DF를 반환.
        [v3.1] 룰북의 모든 '발견된' 성분/함량 데이터를 agg_df에 포함 (엔진 수정됨)
        """
        try:
            # 1. 전처리 (v2.6) - [v3.1] 엔진(v4.9.3)이 모든 성분 함량+브랜드 추출
            agg_df = core_engine.preprocess_data_v2_6(_raw_df.copy(), _rules)
            # 2. 마켓 스코어 계산 (v2.7)
            market_scores = core_engine.calculate_market_score_v2(agg_df, _rules['market_score_weights'])
            agg_df['MARKET_SCORE'] = market_scores
            return agg_df
        except Exception as e:
            st.error(f"델타 데이터 준비 중 오류: {e}")
            return None

    # 룰북을 문자열로 변환하여 캐시 키로 사용 (룰북이 바뀌면 재실행됨)
    rulebook_str = str(rb) 
    delta_df = prepare_delta_data(raw_df, rb)

    # --- [v3.1.2] 오류 수정 로직 ---
    if delta_df is None:
        st.error("델타 분석용 데이터를 준비하지 못했습니다. [Tab 1]의 룰북 설정을 확인하세요.")
    else:
        # delta_df가 None이 아닐 때만 이 블록 실행
        
        # --- [v4.5] A/B 그룹 필터 설정 ---
        st.divider()
        st.subheader("🔬 [A/B] '다중 필터' 설정")
        
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("#### [A 그룹] '비교' 그룹 ")
            with st.container(border=True):
                filters_A = create_filter_box('v4_filters_A', _discovered_rules, delta_df)
            
        with cols[1]:
            st.markdown("#### [B 그룹] '대조' 그룹 ")
            with st.container(border=True):
                b_choice = st.radio(
                    "B그룹 비교 대상:",
                    ["A그룹 외 '그외 제품'", "A그룹 vs '다른 필터'"],
                    key="b_choice",
                    horizontal=True
                )
                
                if b_choice == "A그룹 vs '다른 필터'":
                    filters_B = create_filter_box('v4_filters_B', _discovered_rules, delta_df)
                else:
                    filters_B = None # '그외 제품' 선택

        st.divider()
        st.header(f"🔬 A/B 그룹 분석결과")
        
        # --- [v4.5] A/B 그룹 데이터 정의 ---
        status_col_name = "비교 그룹"
        
        df_A = apply_filters(delta_df, filters_A)
        df_A[status_col_name] = "그룹 A"
        
        if filters_B is not None:
            # "A그룹 vs '다른 필터'"
            df_B = apply_filters(delta_df, filters_B)
            df_B[status_col_name] = "그룹 B"
        else:
            # "A그룹 외 '그외 제품'"
            # (A그룹의 인덱스를 '전체' delta_df에서 제외)
            df_B = delta_df.drop(df_A.index) 
            df_B[status_col_name] = "그룹 B (그 외)"
        
        # '1축 2그림'을 위한 데이터 합치기
        combined_df = pd.concat([df_A, df_B])

        # --- [v4.5 신규] C. '1축 2그림' (Strip Plot) ---
        st.subheader("📈")
        
        # "보유(파랑)/미보유(빨강)" -> A(파랑)/B(빨강)
        color_map = {
            "그룹 A": "blue", 
            "그룹 B": "red", 
            "그룹 B (그 외)": "red"
        }
        
        chart_cols = st.columns(2)
        
        with chart_cols[0]:
            st.markdown("**그림 1: 💲 가격 분포**")
            fig1 = px.strip( # 'Box' -> 'Strip' ("점만 찍기")
                combined_df, 
                x=status_col_name, 
                y='price', 
                color=status_col_name, # 색상 적용
                color_discrete_map=color_map, # "파랑/빨강" 적용
                title="가격(Y) vs A/B 그룹(X)",
                hover_data=['product_name']
            )
            st.plotly_chart(fig1, use_container_width=True)

        with chart_cols[1]:
            st.markdown("**그림 2: 📈 시장 반응 분포**")
            fig2 = px.strip( # 'Box' -> 'Strip' ("점만 찍기")
                combined_df, 
                x=status_col_name, 
                y='MARKET_SCORE', 
                color=status_col_name, # 색상 적용
                color_discrete_map=color_map, # "파랑/빨강" 적용
                title="시장반응(Y) vs A/B 그룹(X)",
                hover_data=['product_name']
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # --- [v4.9.3] D. 원본 제품 목록 ('쭈르륵') (v4.9 '동적 컬럼' 적용) ---
        st.subheader("📋 목록")
        st.caption("('필터'로 사용된 '그 컬럼'의 값들이 '자동으로 추가'되어 'Blackbox'를 제거합니다.)")
        
        list_cols = st.columns(2)
        
        # --- [v4.9.3 신규] '동적 컬럼' 로직 (NameError 수정) ---
        # 1. 'A그룹'에 사용된 필터 '키' 목록 추출
        base_cols = ['product_name', 'price', 'MARKET_SCORE']
        cols_A = base_cols.copy()
        for key in filters_A.keys():
            if key not in cols_A:
                cols_A.append(key)
            # '특수태그'가 필터였다면, 'Blackbox' 제거를 위해 'tags_raw' 추가
            if key in _discovered_rules['tags'] and 'tags_raw' not in cols_A:
                cols_A.append('tags_raw')
        
        # 2. 'B그룹'에 사용된 필터 '키' 목록 추출
        cols_B = base_cols.copy()
        if filters_B is not None: # "다른 필터" 비교 시
            for key in filters_B.keys():
                if key not in cols_B:
                    cols_B.append(key)
                if key in _discovered_rules['tags'] and 'tags_raw' not in cols_B:
                    cols_B.append('tags_raw')
        else: # "그외 제품" 비교 시 (A그룹 필터 컬럼을 동일하게 보여줌)
             cols_B = cols_A.copy()
        # --- [v4.9.3 수정 완료] ---

        with list_cols[0]:
            st.markdown(f"**[A] '비교' 그룹 제품 (n={len(df_A)})**")
            # [v4.9] 'display_cols' -> 'cols_A' (동적 컬럼)
            # (존재하지 않는 컬럼명 오류 방지를 위해, 실제 DF에 있는 컬럼만 필터링)
            valid_cols_A = [col for col in cols_A if col in df_A.columns]
            st.dataframe(df_A[valid_cols_A].sort_values(by='MARKET_SCORE', ascending=False).style.format(precision=1))
            
        with list_cols[1]:
            st.markdown(f"**[B] '대조' 그룹 제품 (n={len(df_B)})**")
            # [v4.9] 'display_cols' -> 'cols_B' (동적 컬럼)
            valid_cols_B = [col for col in cols_B if col in df_B.columns]
            st.dataframe(df_B[valid_cols_B].sort_values(by='MARKET_SCORE', ascending=False).style.format(precision=1))
    # --- [v3.1.2] 오류 수정 'else' 블록 끝 ---