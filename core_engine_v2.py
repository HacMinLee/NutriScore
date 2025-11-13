"""
Project Swan's Eye v2.7.1 (v4.9.3) - Core Engine
- v4.9.3 (사장님 요청): '최종 순위' 표시에 '누락'된 '브랜드'와 'MarketScore'를 '엔진'단에서 '추가'.
    - (1) 'preprocess_data_v2_6'에 '브랜드' (텍스트) 추출 로직 '추가'.
    - (2) 'run_full_analysis_v2_6'에 'MARKET_SCORE'를 'final_df'에 '합류'시키는 로직 '추가'.
- v2.7: '델타 분석기'용 v1.4 'calculate_market_score' 함수 부활
- v2.6.3: '특수태그' 정규식 오류 수정
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import zscore

# ---
# [v2.0] S-Curve 및 Z-Score 유틸리티 (v1.4/v2.0)
# ---

def calculate_z_scores(series, direction='higher_is_better'):
    """ v1.4와 동일 (인덱스 꼬임 버그 수정 버전) """
    z_scores_full = pd.Series(np.nan, index=series.index, dtype=float)
    valid_series = series.dropna()
    
    if not valid_series.empty:
        if valid_series.std() == 0:
            z_scores_valid = pd.Series(0.0, index=valid_series.index)
        else:
            valid_z = zscore(valid_series)
            z_scores_valid = pd.Series(valid_z, index=valid_series.index)
        z_scores_full.update(z_scores_valid)

    final_z_scores = z_scores_full.fillna(0)

    if direction == 'lower_is_better':
        final_z_scores = -final_z_scores
    return final_z_scores

def apply_sigmoid(z_score, k=1.0):
    """ v1.4와 동일 """
    return 100 / (1 + np.exp(-k * z_score))

def calculate_custom_s_curve_score(
    dose, min_dose, rec_dose, rec_score, saturation_factor
):
    """ v2.0에서 설계한 4-파라미터 S-Curve 모델 """
    LOW_SCORE_FLOOR = 5.0 
    MAX_SCORE = 100.0
    
    # 방어 코드 (0으로 나누기, 동일 값)
    if rec_dose == min_dose: rec_dose += 1e-6
    if rec_dose == 0: rec_dose = 1e-6
    
    if pd.isna(dose) or dose < min_dose:
        return 0.0
    elif min_dose <= dose < rec_dose:
        x_norm = (dose - min_dose) / (rec_dose - min_dose)
        k_steepness = 5.0 
        z = k_steepness * (x_norm - 0.5)
        sigmoid_norm = 0.5 * (1.0 + np.tanh(z))
        score_range = rec_score - LOW_SCORE_FLOOR
        score = (sigmoid_norm * score_range) + LOW_SCORE_FLOOR
        return score
    elif dose >= rec_dose:
        score_range = MAX_SCORE - rec_score
        dose_diff = dose - rec_dose
        k = saturation_factor / rec_dose
        additional_score = score_range * (1.0 - np.exp(-k * dose_diff))
        score = rec_score + additional_score
        return min(score, MAX_SCORE)

# ---
# [v2.6] 데이터 전처리 (v1.4 그룹핑 + v2.6 동적 추출)
# ---

def extract_component_value(series, comp_name):
    """
    "성분 : EPA, 함유량 : 690" 같은 텍스트에서 '함유량'을 추출하는 헬퍼 함수
    """
    comp_name_safe = re.escape(comp_name) # 정규식 특수문자 이스케이프
    # "성분", "함유량" 사이의 모든 문자를 허용하고, 공백/쉼표 무시
    pattern = re.compile(
        r"성분\s*:\s*" + comp_name_safe + r"[^,]*,\s*함유량\s*:\s*([\d\.]+)",
        re.IGNORECASE # 대소문자 무시 (혹시 모르니)
    )
    
    for text in series.dropna():
        clean_text = str(text).replace(" ", "") # 공백 제거
        match = pattern.search(clean_text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                continue
    return np.nan

def preprocess_data_v2_6(df, rules):
    """
    [v4.9.3] v1.4의 ffill/groupby 로직과 v2.6의 동적 성분 추출을 결합.
    '제품이름을 주인으로' 설정 + '브랜드' '누락' 복구.
    """
    
    # 1. v1.4의 ffill 로직 (제품명 채우기)
    col_product = rules['columns']['product_name']
    df[col_product] = df[col_product].ffill()
    df = df.dropna(subset=[col_product])
    
    processed_data = []
    grouped = df.groupby(col_product)
    
    # 2. v1.4의 groupby 로직 (제품별 '주렁주렁' 그룹화)
    for product_name, group in grouped:
        product_row = {'product_name': product_name}
        
        # 3. [Score B] 가격, [Market] 리뷰 등 공통 컬럼 추출 (v1.4 방식)
        def safe_get_first(series_str):
            # [v4.9.3] 컬럼이 없는 경우를 대비한 방어 코드
            if series_str not in group.columns:
                return np.nan
            series = group[series_str].dropna()
            if not series.empty:
                # 숫자 추출 로직 (v1.4)
                cleaned = pd.to_numeric(
                    series.astype(str).str.replace(r'[^\d\.]', '', regex=True).replace('', np.nan),
                    errors='coerce'
                ).dropna()
                if not cleaned.empty:
                    return cleaned.iloc[0]
            return np.nan
        
        product_row['price'] = safe_get_first(rules['columns']['price'])
        product_row['review_count'] = safe_get_first(rules['columns']['review_count'])
        product_row['rating'] = safe_get_first(rules['columns']['rating'])
        
        # --- [v4.9.3] '브랜드' '누락' 복구 (텍스트용) ---
        # (룰북에 'brand' 키가 없으면 '브랜드'로 '하드코딩')
        col_brand = rules['columns'].get('brand', '브랜드') 
        if col_brand in group.columns: # 컬럼이 있는지 확인
            series_brand = group[col_brand].dropna()
            if not series_brand.empty:
                product_row['브랜드'] = series_brand.iloc[0] # 텍스트는 [0]번만 가져옴
            else:
                product_row['브랜드'] = np.nan
        else:
            # '브랜드' 컬럼이 룰북에도, CSV에도 없으면 NaN
            product_row['브랜드'] = np.nan
        # --- [v4.9.3 수정 완료] ---

        # 4. [Score A] 핵심성분 동적 추출 (v2.6 핵심)
        col_main_comp = rules['score_a_main_components']['csv_column']
        if col_main_comp in group.columns:
            series_main_comp = group[col_main_comp]
            # [v3.1] 델타 분석기와의 호환성을 위해, '활성화(enabled)' 여부와 관계없이
            #      룰북에 '발견된' 모든 성분의 함량을 우선 추출한다.
            for comp_name in rules['score_a_main_components']['rules'].keys():
                product_row[comp_name] = extract_component_value(series_main_comp, comp_name)
        else:
            # 컬럼이 없으면 모든 성분을 NaN으로 채움
            for comp_name in rules['score_a_main_components']['rules'].keys():
                product_row[comp_name] = np.nan

        # 5. [Score C-1] 보조 성분 동적 추출 (v2.6)
        col_sub_comp = rules['score_c_sub_components']['csv_column']
        if col_sub_comp in group.columns:
            series_sub_comp = group[col_sub_comp]
            # [v3.1] 델타 분석기 호환성을 위해, 모든 보조성분 함량 추출
            for comp_name in rules['score_c_sub_components']['rules'].keys():
                 product_row[comp_name] = extract_component_value(series_sub_comp, comp_name)
        else:
            for comp_name in rules['score_c_sub_components']['rules'].keys():
                product_row[comp_name] = np.nan
        
        # 6. [Score C-2] 특수 태그 텍스트 통째로 가져오기 (v2.6)
        col_tags = rules['score_c_tags']['csv_column']
        if col_tags in group.columns:
            series_tags = group[col_tags].dropna()
            if not series_tags.empty:
                product_row['tags_raw'] = series_tags.iloc[0] # 첫번째 태그 줄만 사용
            else:
                product_row['tags_raw'] = np.nan
        else:
            product_row['tags_raw'] = np.nan
            
        processed_data.append(product_row)

    return pd.DataFrame(processed_data)

# ---
# [v2.6] 스코어링 함수 (v2.0 합산 모델)
# ---

def calculate_score_a(df, rules_dict):
    """ [Score A] 핵심성분 점수 (4-파라미터 S-Curve) """
    component_scores_df = pd.DataFrame(index=df.index)
    total_weight = 0.0
    
    for comp_name, rule in rules_dict['rules'].items():
        if not rule.get('enabled', False): # 'enabled'가 True인 것만 계산
            continue
            
        weight = rule['weight']
        total_weight += weight
        
        # 전처리된 df에서 함량(dose) 데이터 (이미 추출됨)
        doses = df[comp_name] 
        
        comp_score = doses.apply(
            lambda d: calculate_custom_s_curve_score(
                d, rule['min_dose'], rule['rec_dose'], rule['rec_score'], rule['saturation_factor']
            )
        )
        component_scores_df[f'A_{comp_name}'] = comp_score * weight

    if total_weight == 0:
        return pd.Series(0.0, index=df.index), component_scores_df

    final_score_a = component_scores_df.sum(axis=1) / total_weight
    return final_score_a, component_scores_df

def calculate_score_b(df, rules_dict):
    """ [Score B] 가격 점수 (Z-Score) """
    price_z = calculate_z_scores(df['price'], direction='lower_is_better')
    price_score = price_z.apply(lambda z: apply_sigmoid(z, k=rules_dict['k_value']))
    return price_score

def calculate_score_c(df, rules_dict_sub, rules_dict_tags):
    """ [Score C] 보조성분(S-Curve) + 태그(합산) """
    
    # C-1: 보조성분 (S-Curve, Score A와 로직 동일)
    component_scores_df = pd.DataFrame(index=df.index)
    total_weight = 0.0
    
    for comp_name, rule in rules_dict_sub['rules'].items():
        if not rule.get('enabled', False): # 'enabled'가 True인 것만 계산
            continue
            
        weight = rule['weight']
        total_weight += weight
        doses = df[comp_name]
        
        comp_score = doses.apply(
            lambda d: calculate_custom_s_curve_score(
                d, rule['min_dose'], rule['rec_dose'], rule['rec_score'], rule['saturation_factor']
            )
        )
        component_scores_df[f'C1_{comp_name}'] = comp_score * weight
    
    if total_weight == 0:
        score_c1 = pd.Series(0.0, index=df.index)
    else:
        score_c1 = component_scores_df.sum(axis=1) / total_weight

    # C-2: 특수태그 (점수 합산)
    tag_rules = rules_dict_tags['rules']
    series_tags_raw = df['tags_raw']
    score_c2 = pd.Series(0.0, index=df.index)
    
    for tag_name, tag_score in tag_rules.items():
        if pd.isna(tag_name) or tag_score == 0: # 점수가 0이면 무시
            continue
        
        # [v2.6.3] 수정된 정규식
        has_tag = series_tags_raw.str.contains(
            f"{re.escape(tag_name)}\s*\*", na=False, regex=True
        )
        
        score_c2[has_tag] += tag_score

    # C_final = C1점수 * C1비중 + C2점수 * C2비중
    w_c1 = rules_dict_sub['final_weight']
    w_c2 = rules_dict_tags['final_weight']
    total_c_weight = w_c1 + w_c2
    if total_c_weight == 0: total_c_weight = 1.0
    
    final_score_c = ((score_c1 * w_c1) + (score_c2 * w_c2)) / total_c_weight
    
    return final_score_c, score_c1, score_c2, component_scores_df


# ---
# [v2.7 신규] 델타 분석기용 Market Score 계산기 (v1.4 부활)
# ---

def calculate_market_score_v2(agg_df, rules_dict):
    """
    델타 분석기 전용 Market Score를 계산합니다. (v1.4 로직 재활용)
    :param agg_df: 전처리/그룹핑이 완료된 데이터프레임
    :param rules_dict: 'market_score_weights' 룰북 딕셔너리
    :return: (pd.Series) 0~100점의 Market Score
    """
    
    # 룰북에서 가중치와 k값(기울기) 추출
    k_review = rules_dict.get('k_review', 2.0)
    k_rating = rules_dict.get('k_rating', 1.0)
    w_review = rules_dict.get('weight_review', 0.7)
    w_rating = rules_dict.get('weight_rating', 0.3)
    
    # 리뷰 수 (v1.4 로직)
    review_z = calculate_z_scores(agg_df['review_count'], direction='higher_is_better')
    review_score = review_z.apply(lambda z: apply_sigmoid(z, k=k_review))
    
    # 별점 (v1.4 로직)
    rating_z = calculate_z_scores(agg_df['rating'], direction='higher_is_better')
    rating_score = rating_z.apply(lambda z: apply_sigmoid(z, k=k_rating))
    
    # 합산
    total_weight = w_review + w_rating
    if total_weight == 0: total_weight = 1.0
    
    market_score = ( (review_score * w_review) + (rating_score * w_rating) ) / total_weight
    
    return market_score

# ---
# [v4.9.3] 메인 파이프라인 ('MarketScore' '누락' 복구)
# ---

def run_full_analysis_v2_6(df, dynamic_rulebook):
    """ [v4.9.3] v2.6의 모든 분석 파이프라인 총괄 (MarketScore 포함) """
    
    rules = dynamic_rulebook
    
    # 1. 데이터 전처리 (v2.6)
    try:
        agg_df = preprocess_data_v2_6(df.copy(), rules) # (v4.9.3 '브랜드' 포함)
    except KeyError as e:
        # [v4.9.3] 룰북에 'brand'가 추가됐는지 확인하라는 '친절한' [cite: 2025-09-02] 오류 메시지
        if str(e) == "'브랜드'":
             raise ValueError(f"CSV 컬럼 매핑 오류: '브랜드' 컬럼을 CSV에서 찾을 수 없습니다. (main_app의 룰북 'columns'에 'brand':'브랜드'가 '누락'됐는지 확인)")
        raise ValueError(f"CSV 컬럼 매핑 오류: {e} 컬럼을 CSV에서 찾을 수 없습니다. (룰북의 '공통 컬럼' 설정 확인)")
    except Exception as e:
        raise ValueError(f"데이터 전처리 중 오류: {e}")

    # --- [v4.9.3] 'MarketScore' '누락' 복구 (Tab 1 표시용) ---
    market_scores = calculate_market_score_v2(agg_df, rules['market_score_weights'])
    # --- [v4.9.3 수정 완료] ---

    # 2. 엔진별 스코어링 (A, B, C)
    score_a, score_a_details = calculate_score_a(agg_df, rules['score_a_main_components'])
    score_b = calculate_score_b(agg_df, rules['score_b_price'])
    score_c, score_c1, score_c2, score_c_details = calculate_score_c(
        agg_df, 
        rules['score_c_sub_components'],
        rules['score_c_tags']
    )

    # 3. 최종 점수 합산 (최종 가중치)
    w_a = rules['final_weights']['weight_a']
    w_b = rules['final_weights']['weight_b']
    w_c = rules['final_weights']['weight_c']
    total_weight = w_a + w_b + w_c
    if total_weight == 0: total_weight = 1.0
    
    final_score = ( (score_a * w_a) + (score_b * w_b) + (score_c * w_c) ) / total_weight

    # 4. 최종 데이터프레임
    final_df = agg_df.copy()
    final_df['SWAN_SCORE_V2'] = final_score
    final_df['SCORE_A (핵심성분)'] = score_a
    final_df['SCORE_B (가격)'] = score_b
    final_df['SCORE_C (보조/태그)'] = score_c
    
    final_df['MARKET_SCORE'] = market_scores # <-- [v4.9.3] '합류' 코드 "추가"
    
    final_df['C1 (보조성분 점수)'] = score_c1
    final_df['C2 (태그 점수)'] = score_c2
    
    final_df = pd.concat([final_df, score_a_details, score_c_details], axis=1)

    return final_df.sort_values(by='SWAN_SCORE_V2', ascending=False)