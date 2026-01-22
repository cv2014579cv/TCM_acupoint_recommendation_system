import pandas as pd
import numpy as np
import os
import sys

class TcmRecommender:
    def __init__(self, data_path='input_csv'):
        """抓input_csv資料夾裡的 CSV 檔""" 
        self.data_path = data_path
        self._load_data()
        self._init_pair_dict()

    def _load_data(self):
        """載入所有必要的 CSV 檔案"""
        self.match_table = pd.read_csv(os.path.join(self.data_path, 'match_table.csv'))
        self.main_table = pd.read_csv(os.path.join(self.data_path, 'main_table.csv'))
        self.symptom_organ = pd.read_csv(os.path.join(self.data_path, 'symptom_organ.csv'))
        self.meridian_five_element = pd.read_csv(os.path.join(self.data_path, 'meridian_five_element.csv'))
        self.organ_five_element = pd.read_csv(os.path.join(self.data_path, 'organ_five_element.csv'))
        self.intersection_df = pd.read_csv(os.path.join(self.data_path, 'intersection_point.csv'))
        self.meridian_table = pd.read_csv(os.path.join(self.data_path, 'meridian_table.csv'))
        self.meridian_organ = pd.read_csv(os.path.join(self.data_path, 'meridian_organ.csv'))
        
        # 載入座標資料
        self.front_df = pd.read_csv(os.path.join(self.data_path, 'front.csv'))
        self.back_df = pd.read_csv(os.path.join(self.data_path, 'back.csv'))
        self.side_df = pd.read_csv(os.path.join(self.data_path, 'side.csv'))

        # 預處理座標資料，將代碼轉為經脈名稱
        meridian_map = self.meridian_table.set_index('id')['name'].to_dict()
        self.front_df['meridian_name'] = self.front_df['meridian'].map(meridian_map)
        self.back_df['meridian_name'] = self.back_df['meridian'].map(meridian_map)
        self.side_df['meridian_name'] = self.side_df['meridian'].map(meridian_map)

        # 基礎特殊的穴位子集
        # 原穴與俞穴
        self.yuan_df = self.main_table[self.main_table['yuan_primary_point'].notnull()]
        self.yu_df = self.main_table[self.main_table['fiveshu_point'] == '俞穴']
        
        self.yuan_and_yu_df = self.main_table[(self.main_table['yuan_primary_point'].notnull()) & (self.main_table['fiveshu_point'] == '俞穴')]
        self.not_yuan_but_yu_df = self.main_table[(self.main_table['yuan_primary_point'].isnull()) & (self.main_table['fiveshu_point'] == '俞穴')]
        self.not_yu_but_yuan_df = self.main_table[(self.main_table['yuan_primary_point'].notnull()) & (self.main_table['fiveshu_point'] != '俞穴')]
        
        # 非俞穴的五腧穴
        self.not_yu_fiveshu_df = self.main_table[(self.main_table['five_element_point'].notnull()) & (self.main_table['fiveshu_point'] != '俞穴')]

    def _init_pair_dict(self):
        """表裡經對照表"""
        self.pair_dict = {
            '手太陰肺經': '手陽明大腸經', '手陽明大腸經': '手太陰肺經',
            '足陽明胃經': '足太陰脾經', '足太陰脾經': '足陽明胃經',
            '手少陰心經': '手太陽小腸經', '手太陽小腸經': '手少陰心經',
            '足太陽膀胱經': '足少陰腎經', '足少陰腎經': '足太陽膀胱經',
            '手厥陰心包經': '手少陽三焦經', '手少陽三焦經': '手厥陰心包經',
            '足少陽膽經': '足厥陰肝經', '足厥陰肝經': '足少陽膽經'
        }

    def _get_five_element_seq(self, symptoms):
        """計算症狀關聯的五行序列 (n1: 兩症狀同時對應五腧穴貢獻, n2: 臟腑症狀貢獻)"""
        import re
        all_elements = ['木', '火', '土', '金', '水'] # 初始化
        
        def clean_str(s):
            if pd.isna(s): return ""
            return "".join(re.findall(r'[\u4e00-\u9fa5]', str(s)))

        clean_input_symptoms = [clean_str(s) for s in symptoms]
        
        # --- 計算 n1 (基於 A1 集合的五行屬性) ---
        # 1. 找出匹配至少兩個症狀的穴位
        temp_match = self.match_table.copy()
        temp_match['symptom_clean'] = temp_match['symptom'].apply(clean_str)
        matched_acu_counts = temp_match[temp_match['symptom_clean'].isin(clean_input_symptoms)].groupby('acupoint').size()
        a1_acupoints = matched_acu_counts[matched_acu_counts >= 2].index.tolist()
        
        # 過濾掉不是五腧穴的穴位 (確保只有井、滎、俞、經、合穴)
        fiveshu_all_names = set(self.main_table[self.main_table['five_element_point'].notnull()]['acupoint'])
        a1_acupoints = [acu for acu in a1_acupoints if acu in fiveshu_all_names]

        # 2. 取得這些穴位的五行屬性並統計
        a1_info = self.main_table[self.main_table['acupoint'].isin(a1_acupoints)]
        n1_scores = {el: 0 for el in all_elements}
        for _, row in a1_info.iterrows():
            attr = clean_str(row['five_element_point'])
            for el in all_elements:
                if el in attr:
                    n1_scores[el] += 1
        
        # --- 計算 n2 (基於 臟腑-症狀 對應分布) ---
        temp_so = self.symptom_organ.copy()
        temp_so['symptom_clean'] = temp_so['symptom'].apply(clean_str)
        # 取得所有匹配到的 (臟腑, 症狀) 對應
        matched_so = temp_so[temp_so['symptom_clean'].isin(clean_input_symptoms)].copy()
        
        # 統計每個臟腑對應到的「不同症狀」數 (去重後計算)
        matched_so['organ_clean'] = matched_so['organ'].apply(clean_str)
        # 這裡需確每個 (臟腑, 症狀) 對應是唯一的，避免單一症狀因資料庫重複而虛增
        unique_so = matched_so.drop_duplicates(subset=['organ_clean', 'symptom_clean'])
        organ_symptom_counts = unique_so.groupby('organ_clean').size()
        
        # 重要規則：僅保留匹配至少同時對應兩個症狀的臟腑 (Threshold >= 2)
        valid_organ_counts = organ_symptom_counts[organ_symptom_counts >= 2].to_dict()
        
        # 將臟腑權重歸納至五行
        org_five_map = self.organ_five_element.copy()
        org_five_map['organ_clean'] = org_five_map['organ'].apply(clean_str)
        org_five_map['five_clean'] = org_five_map['five_element'].apply(clean_str)
        
        n2_scores = {el: 0 for el in all_elements}
        for org, count in valid_organ_counts.items():
            # 找出該臟腑對應的五行
            target_el_row = org_five_map[org_five_map['organ_clean'] == org]
            if not target_el_row.empty:
                element_str = target_el_row.iloc[0]['five_clean']
                for el in all_elements:
                    if el in element_str:
                        n2_scores[el] += count
                        
        # --- 整合與排序 ---
        # 1. 第一次排序 (基於 n1)
        n1_data = []
        base_order_map = {el: i for i, el in enumerate(all_elements)}
        for el in all_elements:
            n1_data.append({'element': el, 'n1': n1_scores[el], 'base_rank': base_order_map[el]})
        
        n1_df = pd.DataFrame(n1_data)
        n1_sorted_elements = n1_df.sort_values(by=['n1', 'base_rank'], ascending=[False, True])['element'].tolist()
        n1_rank_map = {el: i for i, el in enumerate(n1_sorted_elements)}

        # 2. 第二次排序 (基於 n1 + n2, 並以 n1 排序結果為 tie-breaker)
        importance_data = []
        for el in all_elements:
            importance_data.append({
                'element': el, 
                'importance': n1_scores[el] + n2_scores[el],
                'n1_rank': n1_rank_map[el]
            })
            
        importance_df = pd.DataFrame(importance_data)
        
        final_seq = importance_df.sort_values(by=['importance', 'n1_rank'], ascending=[False, True])['element'].tolist()
        return final_seq, importance_df

    def _get_meridian_hier_seq(self, a1_df, five_element_seq):
        """計算經脈層級序列 (基於 A1 候選穴位)"""
        if a1_df.empty:
            return []
            
        meridian_counts = a1_df.groupby('meridian_name').size().to_dict()
        return b_meridian_seq, b_five_seq

    def _get_b_meridian_and_five_seq(self, b1_df):
        """計算 B 集專屬的經脈序列與五行序列 (基於 B1 穴位頻率)"""
        if b1_df.empty:
            return [], []
            
        # 1. B1_acupoint_count (對應 Notebook: grouping and sorting)
        b1_acupoint_count = b1_df.groupby('meridian_name').size().reset_index(name='number_of_acupoint')
        b1_acupoint_count = b1_acupoint_count.sort_values('number_of_acupoint', ascending=False)
        
        # 2. B1_meridian_list (對應 Notebook)
        b_mer_seq = b1_acupoint_count['meridian_name'].tolist()
        
        # 3. B1_five_importance (對應 Notebook: merge with meridian_five_df)
        meridian_five_df = self.meridian_five_element.rename(columns={'meridian': 'meridian_name'})
        b1_five_importance = b1_acupoint_count.merge(meridian_five_df, on='meridian_name', how='left')
        
        # 4. 生成 B 之五行序列 (b_five_seq)
        # 依照 B1 經脈重要度順序提取五行屬性
        b_five_seq = []
        for el in b1_five_importance['five_element']:
            if pd.notna(el):
                el_str = str(el).strip()
                if f"{el_str}穴" not in b_five_seq:
                    b_five_seq.append(f"{el_str}穴")
        
        return b_mer_seq, b_five_seq

    def _get_special_candidates(self, meridian_list, five_element_seq, is_path_a=True, intersection_targets=None):
        """獲取特定經脈的特殊穴位 (原/俞/五腧/交會)"""
        if not meridian_list:
            return pd.DataFrame()
            
        # 1. 基礎篩選：獲取該經脈的所有特殊穴位
        subset_yuan_yu = self.yuan_and_yu_df[self.yuan_and_yu_df['meridian_name'].isin(meridian_list)]
        subset_not_yuan_yu = self.not_yuan_but_yu_df[self.not_yuan_but_yu_df['meridian_name'].isin(meridian_list)]
        subset_not_yu_yuan = self.not_yu_but_yuan_df[self.not_yu_but_yuan_df['meridian_name'].isin(meridian_list)]
        fiveshu_cands = self.not_yu_fiveshu_df[self.not_yu_fiveshu_df['meridian_name'].isin(meridian_list)]
        acu_attr_map = self.main_table.set_index('acupoint')['five_element_point'].to_dict()
        
        if not fiveshu_cands.empty:
            fiveshu_cands = fiveshu_cands.copy()
            clean_seq = [s.strip().replace('穴', '') for s in five_element_seq]
            mer_five_map = self.meridian_five_element.set_index('meridian')['five_element'].to_dict()
            
            if is_path_a:
                # 路徑 A (症狀路徑)：遵循五行序列優先序
                def get_priority(val):
                    name = str(val).strip().replace('穴', '')
                    return clean_seq.index(name) if name in clean_seq else 99
                
                # 僅保留在序列中的五腧穴 (或俞穴/原穴)
                fiveshu_cands['priority'] = fiveshu_cands['five_element_point'].apply(get_priority)
                # 注意：第一標的或在序列中的點保留
                def path_a_filter(row):
                    m_name = row['meridian_name']
                    try:
                        m_idx = meridian_list.index(m_name)
                    except ValueError: return False
                    
                    # 規則 B：次要經脈僅保留原穴與俞穴
                    if m_idx > 0:
                        return row['fiveshu_point'] == '俞穴' or pd.notnull(row['yuan_primary_point'])
                    
                    # 第一標的：保留原穴、俞穴或在五行序列中的點
                    is_yu_yuan = row['fiveshu_point'] == '俞穴' or pd.notnull(row['yuan_primary_point'])
                    if is_yu_yuan: return True
                    
                    val = row.get('five_element_point')
                    if pd.isna(val): val = acu_attr_map.get(row['acupoint'])
                    el = str(val).strip().replace('穴', '') if pd.notnull(val) else ""
                    return el in clean_seq
                            
                subset_yuan_yu = subset_yuan_yu[subset_yuan_yu.apply(path_a_filter, axis=1)]
                subset_not_yuan_yu = subset_not_yuan_yu[subset_not_yuan_yu.apply(path_a_filter, axis=1)]
                subset_not_yu_yuan = subset_not_yu_yuan[subset_not_yu_yuan.apply(path_a_filter, axis=1)]
                fiveshu_cands = fiveshu_cands[fiveshu_cands.apply(path_a_filter, axis=1)]
            else:
                # 路徑 B (點擊路徑)
                if len(meridian_list) > 0:
                    def path_b_filter(row):
                        m_name = row['meridian_name']
                        try:
                            m_idx = meridian_list.index(m_name)
                        except ValueError: return False

                        if m_idx == 0:
                            # B2 第一名：保留原穴、俞穴，以及順位序列中的穴位
                            is_yu_yuan = row['fiveshu_point'] == '俞穴' or pd.notnull(row['yuan_primary_point'])
                            val = row.get('five_element_point')
                            if pd.isna(val): val = acu_attr_map.get(row['acupoint'])
                            el = str(val).strip().replace('穴', '') if pd.notnull(val) else ""
                            return is_yu_yuan or el in clean_seq
                        else:
                            # B2 第二名：僅保留原穴與俞穴
                            return row['fiveshu_point'] == '俞穴' or pd.notnull(row['yuan_primary_point'])

                    subset_yuan_yu = subset_yuan_yu[subset_yuan_yu.apply(path_b_filter, axis=1)]
                    subset_not_yuan_yu = subset_not_yuan_yu[subset_not_yuan_yu.apply(path_b_filter, axis=1)]
                    subset_not_yu_yuan = subset_not_yu_yuan[subset_not_yu_yuan.apply(path_b_filter, axis=1)]
                    fiveshu_cands = fiveshu_cands[fiveshu_cands.apply(path_b_filter, axis=1)]
        
        # 3. 交會穴篩選 (正式採用 Notebook 核心邏輯修正版)
        # 3.1 準備基礎資料結構 (比照 Notebook)
        # 確保從 CSV 讀取時的順序被保留
        temp_inter = self.intersection_df.copy()
        temp_inter['csv_order'] = range(len(temp_inter))
        intersection_dict = temp_inter.sort_values('csv_order').groupby('intersect_meridian')['acupoint'].apply(list).to_dict()
        main_table_dict = self.main_table[['acupoint', 'meridian_name']].set_index('acupoint').to_dict(orient='index')
        
        # 決定目標序列 (A2/B2 前二順位)
        paired_meridians = intersection_targets if (intersection_targets and len(intersection_targets) > 0) else meridian_list
        
        # 3.2 構建初選名單：根據表裡經的經脈序列，從 intersection_dict 獲取對應的共同穴位
        intersect_candidate_data = []
        for meridian in paired_meridians:
            acupoints = intersection_dict.get(meridian, [])
            for acupoint in acupoints:
                intersect_candidate_data.append({'acupoint': acupoint, 'meridian_name': meridian})
        
        intersect_candidate_df = pd.DataFrame(intersect_candidate_data, columns=['acupoint', 'meridian_name'])
        
        if intersect_candidate_df.empty:
            inter_info = pd.DataFrame()
        else:
            # 3.3 刷新歸屬：根據 main_table_dict 將穴位刷新為其直屬經脈
            result_rows = []
            for _, row in intersect_candidate_df.iterrows():
                acu = row['acupoint']
                if acu in main_table_dict:
                    real_mer = main_table_dict[acu]['meridian_name']
                    result_rows.append({'acupoint': acu, 'meridian_name': real_mer})
            
            result_df = pd.DataFrame(result_rows)
            
            # 3.4 篩選符合目標序列的經脈
            result_df = result_df[result_df['meridian_name'].isin(paired_meridians)].copy()
            
            if result_df.empty:
                inter_info = pd.DataFrame()
            else:
                # 3.5 計算交會數與排序
                result_df['count'] = result_df.groupby('acupoint')['acupoint'].transform('count')
                mer_order_map = {m: i for i, m in enumerate(paired_meridians)}
                result_df['meridian_order'] = result_df['meridian_name'].map(mer_order_map)
                
                # 按穴位數量由多至少排 (count)，同樣數量的穴位再按 paired_meridians 順序排列 (meridian_order)
                result_df = result_df.sort_values(by=['count', 'meridian_order'], ascending=[False, True])
                
                # 去除重複穴位
                result_df = result_df.drop_duplicates(subset='acupoint')
                
                # 3.6 核心過濾規則 (Rule B)
                # 交會穴過濾：先取 count 為 2 (同時交會兩經脈) 的穴位，再把直屬經脈是第二順位表裡經的交會穴濾掉
                max_c = result_df['count'].max()
                if max_c > 1:
                    result_df = result_df[result_df['count'] > 1]
                    # 過濾掉第二順位的經絡
                    if len(paired_meridians) > 1:
                        second_meridian = paired_meridians[1]
                        result_df = result_df[result_df['meridian_name'] != second_meridian]
                elif max_c == 1 and len(paired_meridians) > 1:
                    # 如果最高交會數只有1口，但目標經脈不只一條，依照規則清空不取
                    result_df = pd.DataFrame(columns=result_df.columns)
                # else: max==1 and len==1, 保持全取
                
                if result_df.empty:
                    inter_info = pd.DataFrame()
                else:
                    # 3.7 回填完整的 main_table 資料，並保持 result_df 的原始順序
                    # 使用 merge 確保屬性補齊的同時，順序跟著初選名單走
                    inter_info = result_df[['acupoint']].merge(self.main_table, on='acupoint', how='left')

        return pd.concat([subset_yuan_yu, subset_not_yuan_yu, subset_not_yu_yuan, fiveshu_cands, inter_info]).drop_duplicates(subset='acupoint').reset_index(drop=True)

    def get_a_set(self, symptoms):
        """基於症狀輸入推薦 (A1)"""
        if not symptoms:
            return pd.DataFrame(), {}, []
            
        # A1 初步篩選 (字串清洗)
        def clean_str(s):
            if pd.isna(s): return ""
            import re
            return "".join(re.findall(r'[\u4e00-\u9fa5]', str(s)))
            
        clean_input = [clean_str(s) for s in symptoms]
        temp_match = self.match_table.copy()
        temp_match['symptom_clean'] = temp_match['symptom'].apply(clean_str)
        
        # 1. 取得五腧穴/原俞穴清單作為 A1 候選
        special_acus = set(self.not_yu_fiveshu_df['acupoint']) | set(self.yuan_and_yu_df['acupoint']) | \
                       set(self.not_yuan_but_yu_df['acupoint']) | set(self.not_yu_but_yuan_df['acupoint'])
        
        # 2. 統計每個穴位匹配到的症狀集合
        matched_acu = temp_match[temp_match['symptom_clean'].isin(clean_input)]
        matched_acu_special = matched_acu[matched_acu['acupoint'].isin(special_acus)].copy()
        acu_sym_map = matched_acu_special.groupby('acupoint')['symptom_clean'].apply(set).to_dict()
        
        # 3. 過濾：至少匹配 2 個症狀
        a1_cands = [acu for acu, syms in acu_sym_map.items() if len(syms) >= 2]
        a1_df = self.main_table[self.main_table['acupoint'].isin(a1_cands)].copy()
        a1_df['match_count'] = a1_df['acupoint'].apply(lambda x: len(acu_sym_map[x]))
        a1_df = a1_df[a1_df['five_element_point'].notnull()]
        
        if a1_df.empty:
            return pd.DataFrame(), {}, []
            
        # 4. 構建層級序列 (Hierarchical Logic)
        path1_matches = []
        for acu in a1_df['acupoint']:
            for s in acu_sym_map[acu]:
                path1_matches.append({'meridian_name': a1_df[a1_df['acupoint']==acu]['meridian_name'].iloc[0], 'symptom': s})
        
        path1_df = pd.DataFrame(path1_matches).drop_duplicates()
        importance_a_scores = path1_df.groupby('meridian_name')['symptom'].nunique().reset_index()
        importance_a_scores.columns = ['meridian_name', 'symptom_count']
        
        grouped = importance_a_scores.groupby('symptom_count')['meridian_name'].apply(list).reset_index()
        grouped = grouped.sort_values(by='symptom_count', ascending=False)
        hier_lists = grouped['meridian_name'].tolist()
        
        hier_flattened = []
        for sublist in hier_lists:
            for m in sublist:
                if m not in hier_flattened:
                    hier_flattened.append(m)
        
        importance_a_dict = importance_a_scores.set_index('meridian_name')['symptom_count'].to_dict()
        
        return a1_df, importance_a_dict, hier_flattened

    def get_b_set(self, click_x, click_y, view):
        """基於點擊輸入推薦 (B1)"""
        # 判定是否在關節區
        range_val = 25
        joint_regions = {
            'front': [
                (240, 390, 15, 250), (165, 265, 210, 380), (365, 455, 210, 380),
                (10, 135, 590, 750), (475, 600, 620, 760), (220, 395, 595, 730),
                (185, 290, 855, 975), (310, 410, 855, 975), (165, 290, 1130, 1260), (310, 425, 1130, 1260)
            ],
            'back': [
                (235, 400, 8, 265), (135, 295, 190, 335), (345, 505, 190, 335),
                (25, 145, 600, 765), (485, 600, 600, 775), (200, 295, 890, 1000),
                (315, 410, 890, 1000), (140, 290, 1165, 1265), (310, 450, 1165, 1265)
            ],
            'side': [
                (265, 425, 5, 250), (310, 370, 490, 580), (360, 455, 810, 920),
                (275, 395, 1060, 1140), (285, 465, 1100, 1205)
            ]
        }
        
        for (x1, x2, y1, y2) in joint_regions.get(view, []):
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                range_val = 10
                break
        
        # B1 (直屬) 篩選
        df_view = getattr(self, f"{view}_df")
        b1_matched = df_view[
            (df_view['x'] >= click_x - range_val) & (df_view['x'] <= click_x + range_val) &
            (df_view['y'] >= click_y - range_val) & (df_view['y'] <= click_y + range_val)
        ]
        
        b1_df = self.main_table[self.main_table['acupoint'].isin(b1_matched['acupoint'])].copy()
        # B1 匹配次數設為 1 (或 0.5) 以便排在 A1 之後，但在 B2 之前
        b1_df['match_count'] = 1
        return b1_df

    def union_and_sort(self, a1_df, b1_df, a2_df, b2_df, five_seq, prioritized_mer_seq=None):
        """合併 A, B 路徑結果並進行全局排序"""
        # --- 1. 直屬經脈聯集 (A1 + B1) ---
        direct_union = pd.concat([a1_df, b1_df]).drop_duplicates(subset='acupoint')
        if direct_union.empty:
            direct_final = pd.DataFrame()
        else:
            # 計算直屬出現頻率 (用於輔助分組)
            meridian_counts = direct_union.groupby('meridian_name').size().to_dict()
            direct_union['number_of_acupoint'] = direct_union['meridian_name'].map(meridian_counts)
            
            meridian_base_order = ['手少陽三焦經', '手厥陰心包經', '手太陰肺經', '手陽明大腸經', '足太陰脾經', '足陽明胃經', '手少陰心經', '手太陽小腸經', '足太陽膀胱經', '足少陰腎經', '足少陽膽經', '足厥陰肝經']
            meridian_five = self.meridian_five_element.set_index('meridian')['five_element'].to_dict()
            clean_five_seq = [s.strip().replace('穴', '') for s in five_seq]

            # 核心經脈權重序列
            if prioritized_mer_seq:
                unique_mers = prioritized_mer_seq
            else:
                unique_mers = list(meridian_counts.keys())
                def get_meridian_rank_key(m):
                    el = meridian_five.get(m, "").strip().replace('穴', '')
                    f_rank = clean_five_seq.index(el) if el in clean_five_seq else 99
                    cnt = meridian_counts.get(m, 0)
                    return (-cnt, f_rank, meridian_base_order.index(m) if m in meridian_base_order else 99)
                unique_mers.sort(key=get_meridian_rank_key)
            
            # --- 2. 直屬排序 (按匹配次數分組，實現組內交錯) ---
            direct_final = self._sort_logic_by_group(direct_union, 'number_of_acupoint', five_seq, ascending=False, unique_mers=unique_mers)

        # --- 3. 表裡經脈聯集與排序 (A2 + B2) (嚴格比照 Notebook 邏輯) ---
        # 求表裡經脈穴位聯集
        paired_union = pd.concat([a2_df, b2_df], ignore_index=True)
        
        # 規則：表裡經穴位不可與直屬經脈穴位重疊
        if not direct_final.empty and not paired_union.empty:
            direct_acupoints = set(direct_final['acupoint'])
            paired_union = paired_union[~paired_union['acupoint'].isin(direct_acupoints)]

        if paired_union.empty:
            paired_final = pd.DataFrame()
        else:
            # 3.1 求聯集表裡經序列 (get_paired_meridians 邏輯)
            paired_mer_base = []
            for dm in prioritized_mer_seq:
                if dm in self.pair_dict:
                    pm = self.pair_dict[dm]
                    if pm not in paired_mer_base:
                        paired_mer_base.append(pm)
            
            # 3.2 依照表裡經脈序列、穴位類別與五行序列排序 (sort_paired_df 邏輯)
            # 將經脈序列轉為 DataFrame，並添加排序順序
            meridian_order_df = pd.DataFrame({
                'meridian_name': paired_mer_base,
                'sort_order': range(len(paired_mer_base))
            })
            
            # 合併 DataFrame
            merged_df = pd.merge(paired_union, meridian_order_df, on='meridian_name', how='left')
            # 僅保留在序列中的經脈穴位
            merged_df = merged_df[merged_df['sort_order'].notnull()]
            
            if merged_df.empty:
                paired_final = pd.DataFrame()
            else:
                # 對每個 sort_order 分組進行處理 (對應 Notebook 的 grouped 處理)
                merged_df = merged_df.sort_values(by='sort_order', ascending=True)
                grouped = merged_df.groupby('sort_order', sort=False)
                sorted_dfs = []
                
                for _, group in grouped:
                    # 對組內穴位執行「穴位類別與五行排序」
                    # 傳入 group_key=None 表示僅對該組進行類別/五行排序
                    sorted_group = self._sort_logic_by_group(group, None, five_seq)
                    sorted_dfs.append(sorted_group)
                
                # 合併排序好的子 DataFrame
                paired_final = pd.concat(sorted_dfs)
                
                # 移除臨時欄位 sort_order
                if 'sort_order' in paired_final.columns:
                    paired_final = paired_final.drop(columns=['sort_order'])
                
                # 去除重複的穴位
                paired_final = paired_final.drop_duplicates(subset='acupoint').reset_index(drop=True)
        
        return direct_final, paired_final

    def _sort_logic_by_group(self, df, group_col, five_seq, ascending=True, unique_mers=None):
        """分組穴位排序邏輯 (同組內按類別與五行排序)"""
        if df.empty: return df
        
        acu_attr_map = self.main_table.set_index('acupoint')['five_element_point'].to_dict()
        yuan_yu_set = set(self.yuan_and_yu_df['acupoint'])
        yu_only_set = set(self.not_yuan_but_yu_df['acupoint'])
        yuan_only_set = set(self.not_yu_but_yuan_df['acupoint'])
        fiveshu_set = set(self.not_yu_fiveshu_df['acupoint'])
        inter_set = set(self.intersection_df['acupoint'])
        clean_five_seq = [s.strip().replace('穴', '') for s in five_seq]

        def get_type_rank(row):
            acu = row['acupoint']
            if acu in yuan_yu_set: return 1
            if acu in yu_only_set: return 2
            if acu in yuan_only_set: return 3
            if acu in fiveshu_set: return 4
            if acu in inter_set: return 5
            return 99

        def get_five_rank(row):
            acu = row['acupoint']
            val = row.get('five_element_point')
            if pd.isna(val) or val is None: val = acu_attr_map.get(acu)
            if pd.isna(val) or val is None:
                mer = row['meridian_name']
                if acu in yu_only_set:
                    return clean_five_seq.index('木') if '陽' in mer and '木' in clean_five_seq else (clean_five_seq.index('土') if '土' in clean_five_seq else 99)
                return 99
            element_name = str(val).strip().replace('穴', '')
            try: return clean_five_seq.index(element_name)
            except ValueError: return 99

        # 核心排序：先按分組欄位排序，組內再按類別、經脈序與五行屬性順序排序
        if unique_mers is None:
            meridian_counts = df.groupby('meridian_name').size().to_dict()
            meridian_base_order = ['手少陽三焦經', '手厥陰心包經', '手太陰肺經', '手陽明大腸經', '足太陰脾經', '足陽明胃經', '手少陰心經', '手太陽小腸經', '足太陽膀胱經', '足少陰腎經', '足少陽膽經', '足厥陰肝經']
            meridian_five = self.meridian_five_element.set_index('meridian')['five_element'].to_dict()
            
            unique_mers = list(meridian_counts.keys())
            def get_stable_mer_rank(m):
                el = meridian_five.get(m, "").strip().replace('穴', '')
                f_rank = clean_five_seq.index(el) if el in clean_five_seq else 99
                cnt = meridian_counts.get(m, 0)
                return (-cnt, f_rank, meridian_base_order.index(m) if m in meridian_base_order else 99)
            unique_mers.sort(key=get_stable_mer_rank)

        if group_col is None:
            unique_groups = [None]
        elif group_col == 'm_order':
            # m_order 是經脈優先序索引，應由小到大 (0, 1, 2...)
            unique_groups = sorted(df[group_col].unique())
        elif group_col == 'number_of_acupoint':
            # 這是匹配次數分組，應由大到小
            unique_groups = sorted(df[group_col].unique(), reverse=True)
        else:
            unique_groups = sorted(df[group_col].unique(), reverse=(not ascending))
            
        sorted_dfs = []
        for g_val in unique_groups:
            group_df = df[df[group_col] == g_val].copy() if group_col is not None else df.copy()
            # 獲取組內經脈的優先權位
            group_df['meridian_rank'] = group_df['meridian_name'].apply(lambda x: unique_mers.index(x) if x in unique_mers else 99)
            group_df['type_rank'] = group_df.apply(get_type_rank, axis=1)
            group_df['five_rank'] = group_df.apply(get_five_rank, axis=1)
            group_df['original_idx'] = group_df.index # 紀錄原始 CSV 索引位位置
            
            # 排序：1. 穴位型態 2. 五行屬性位階 3. 經脈重要序列 
            # (不顯式排序索引，維持併表後的自然穩定順序，以對齊 Notebook 結果)
            group_df = group_df.sort_values(
                ['type_rank', 'five_rank', 'meridian_rank'], 
                ascending=[True, True, True]
            )
            sorted_dfs.append(group_df.drop(columns=['type_rank', 'five_rank', 'meridian_rank', 'original_idx']))
            
        return pd.concat(sorted_dfs).reset_index(drop=True)

    def recommend(self, symptoms, click_pos=None):
        """外部調用接口"""
        # 1. 獲取直屬經脈 (Path A + Path B)
        a1_df, a_mer_scores, hier_flattened = self.get_a_set(symptoms)
        five_seq, _ = self._get_five_element_seq(symptoms) if symptoms else ([], None)
        
        if click_pos:
            b1_df = self.get_b_set(click_pos['x'], click_pos['y'], click_pos['view'])
            b_mer_hier, b_five_seq = self._get_b_meridian_and_five_seq(b1_df)
            b_mer_scores = b1_df.groupby('meridian_name').size().to_dict()
        else:
            b1_df = pd.DataFrame()
            b_mer_hier, b_five_seq = [], []
            b_mer_scores = {}

        # 2. 定義基準與五行排序
        meridian_five = self.meridian_five_element.set_index('meridian')['five_element'].to_dict()
        clean_five = [s.strip().replace('穴', '') for s in five_seq]
        
        # 3. 按照各自路徑序列選擇 A2/B2 標的
        # A2: 使用重要度 A + 層級序列排名
        a1_mer_list = list(a1_df['meridian_name'].unique()) if not a1_df.empty else []
        def a1_rank_key(m):
            score = a_mer_scores.get(m, 0)
            el = meridian_five.get(m, "").strip().replace('穴', '')
            f_rank = clean_five.index(el) if el in clean_five else 99
            h_rank = hier_flattened.index(m) if m in hier_flattened else 99
            return (-score, f_rank, h_rank)
        
        a1_mer_ranked = sorted(a1_mer_list, key=a1_rank_key)
        a_all_pairs = [self.pair_dict[m] for m in a1_mer_ranked if m in self.pair_dict]
        a1_top_2_pairs = a_all_pairs[:2]
        # A2 僅從前二名表裡經脈中選穴
        a2_df = self._get_special_candidates(a1_top_2_pairs, five_seq, is_path_a=True, intersection_targets=a1_top_2_pairs)
        
        # B2: 使用重要度 B
        if not b1_df.empty:
            # 直接使用由 _get_b_meridian_and_five_seq 生成的序列，對齊 Notebook 邏輯
            b1_mer_ranked = b_mer_hier
            
            b_all_pairs = [self.pair_dict[m] for m in b1_mer_ranked if m in self.pair_dict]
            b1_top_2_pairs = b_all_pairs[:2]
            # B2 僅從前二名表裡經脈中選穴
            b2_df = self._get_special_candidates(b1_top_2_pairs, b_five_seq, is_path_a=False, intersection_targets=b1_top_2_pairs)
        else:
            b2_df = pd.DataFrame()
            
        # 4. 全域聯集排序 (依照 Notebook 累加計數邏輯)
        affiliated_df = pd.concat([a1_df, b1_df], ignore_index=True)
        if not affiliated_df.empty:
            counts_df = affiliated_df.groupby('meridian_name')['acupoint'].count().reset_index(name='cnt')
            counts_df = counts_df.sort_values('cnt', ascending=False)
            all_mers_ranked = counts_df['meridian_name'].tolist()
        else:
            all_mers_ranked = []
            
        direct_final, paired_final = self.union_and_sort(a1_df, b1_df, a2_df, b2_df, five_seq, all_mers_ranked)
        
        # 清理資料：將 Pandas 的 NaN 轉換為空字串，以避免 JSON 序列化錯誤
        direct_final = direct_final.fillna('')
        paired_final = paired_final.fillna('')
        
        return {
            'direct_recommendations': direct_final.to_dict('records'),
            'paired_recommendations': paired_final.to_dict('records')
        }

    def get_all_symptoms(self):
        """
        獲取資料庫中所有不重複的症狀清單
        用途：供前端下拉選單使用，提升使用者輸入體驗
        """
        if self.match_table is not None:
            # 從對照表中提取所有症狀名稱，去除重複並按筆劃/字母排序
            return sorted(self.match_table['symptom'].unique().tolist())
        return []
