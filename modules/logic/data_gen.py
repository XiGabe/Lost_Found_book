import random
import string
import csv
import copy
import os

# ==========================================
# 配置与常量
# ==========================================
OUTPUT_FILE = "data/synthetic_pairs/lcc_training_data.csv"
NUM_SAMPLES = 300000

PREFIXES = ["OLIN", "URIS", "KROCH", "MANN", "LAW", "MATH", "FINE ARTS"]
PREFIX_MUTATIONS = {
    "OLIN": ["QLIN", "DLIN", "0LIN", "OLI"],
    "URIS": ["UR1S", "UR IS", "JRIS"],
    "BV": ["3V", "B V"],
    "PR": ["P R", "PIR"],
    "PQ": ["P Q", "PO"]
}

SUFFIX_TYPES = ['v.', 'no.', 'c.', 'copy', 'vol.', 'Bd.', 'T.', 'Heft', 'suppl.', 'pt.', '+', '++']

GARBAGE_TEXTS = ["LeoS. Olschki", "JOHN DONALD", "EKAOEEIT ANOE", "CSIC", "PEETERS", "ISBN 0-8298-0944-9"]

# 依然保留混淆字典，但大幅降低全局触发率
OCR_CONFUSION = {
    '0': ['O', 'D'], '1': ['I', 'l', '|'], '2': ['Z', '7'], '3': ['E', 'B', '8'], 
    '4': ['A', 'H'], '5': ['S', '6'], '6': ['G', 'b'], '8': ['B', '3'], 
    '9': ['g', 'q'], 'B': ['8', 'E'], 'D': ['0', 'O'], 'Z': ['2'], '.': [',', '-', " "]
}

class LCCCallNumber:
    def __init__(self):
        self.cls_letters = self._gen_letters()
        self.cls_number = random.randint(1, 9999)
        self.has_cls_decimal = random.random() > 0.4
        self.cls_decimal = str(random.randint(1, 999)) if self.has_cls_decimal else ""
        
        self.cutter1_let = random.choice(string.ascii_uppercase)
        self.cutter1_num = str(random.randint(2, 999))
        # 【新增】盲区1：15% 概率附加工作号 (Work Mark)
        self.cutter1_workmark = random.choice(['x', 'X', 'z', 'Z', 'W', 'c', 'd']) if random.random() < 0.15 else ""
        
        self.has_cutter2 = random.random() > 0.4
        self.cutter2_let = random.choice(string.ascii_uppercase) if self.has_cutter2 else ""
        self.cutter2_num = str(random.randint(2, 999)) if self.has_cutter2 else ""
        # 【新增】盲区1：第二 Cutter 也可能有工作号
        self.cutter2_workmark = random.choice(['x', 'X', 'z', 'Z', 'W', 'c', 'd']) if self.has_cutter2 and random.random() < 0.1 else ""
        
        self.has_year = random.random() > 0.5
        self.year = random.randint(1880, 2025) if self.has_year else 0
        # 【新增】盲区2：10% 概率附加版本号 (Year Suffix)
        self.year_suffix = random.choice(['a', 'b', 'c', 'z']) if self.has_year and random.random() < 0.1 else ""
        
        self.has_suffix = random.random() > 0.8
        # 【新增】盲区4：引入大小写变异
        suffix_pool = ['v.', 'V.', 'no.', 'No.', 'c.', 'C.', 'copy', 'vol.', 'Bd.', 'T.', 't.', 'Heft', 'suppl.', 'pt.', 'Pt.']
        self.suffix_type = random.choice(suffix_pool) if self.has_suffix else ""
        self.suffix_num = random.randint(1, 50) if self.has_suffix else 0
        
        # 【新增】盲区4：10% 概率生成复合后缀 (例如 v.2 pt.1)
        self.has_sub_suffix = self.has_suffix and random.random() < 0.1
        self.sub_suffix_type = random.choice(['pt.', 'Pt.', 'no.']) if self.has_sub_suffix else ""
        self.sub_suffix_num = random.randint(1, 10) if self.has_sub_suffix else 0

        # 【新增】盲区3：超大本标识的游离位置
        self.oversize_mark = ""
        if random.random() < 0.05:
            self.oversize_mark = random.choice(['+', '++'])

    def _gen_letters(self):
        length = random.choices([1, 2, 3], weights=[0.2, 0.7, 0.1])[0]
        return ''.join(random.choices(string.ascii_uppercase, k=length))

    def to_string(self):
        parts = []
        cls_str = f"{self.cls_letters} {self.cls_number}"
        if self.has_cls_decimal: cls_str += f".{self.cls_decimal}"
        parts.append(cls_str)
        
        # 拼接带工作号的 Cutter
        parts.append(f".{self.cutter1_let}{self.cutter1_num}{self.cutter1_workmark}")
        if self.has_cutter2: 
            parts.append(f"{self.cutter2_let}{self.cutter2_num}{self.cutter2_workmark}")
        
        # 【新增】盲区3：模拟 '+' 出现在 Cutter 之后，年份之前
        if self.oversize_mark and random.random() < 0.3:
            parts[-1] += self.oversize_mark
            self.oversize_mark = "" # 已经使用，清空
            
        if self.has_year: 
            parts.append(f"{self.year}{self.year_suffix}")
            
        if self.has_suffix:
            suffix_str = f"{self.suffix_type}{self.suffix_num}"
            if self.has_sub_suffix:
                suffix_str += f" {self.sub_suffix_type}{self.sub_suffix_num}"
            parts.append(suffix_str)
            
        # 【新增】盲区3：模拟 '+' 出现在结尾
        if self.oversize_mark:
            parts.append(self.oversize_mark)
            
        return "\n".join(parts)

# ==========================================
# V3.3 专项攻击逻辑 (修复与增强)
# ==========================================
def generate_v33_hard_pair():
    """生成高度相似但语义不同的 A < B"""
    A = LCCCallNumber()
    B = copy.deepcopy(A)

    attack_type = random.choices(
        ['cutter_letter', 'cutter_digit_len', 'cutter_digit_val', 'year_near',
         'decimal_diff', 'cutter2_diff', 'volume_diff', 'main_class_anchor',
         'minimal_diff'],  # 新增：极小差异硬负样本
        weights=[0.12, 0.12, 0.08, 0.08, 0.08, 0.08, 0.04, 0.20, 0.20]  # minimal_diff 占 20%
    )[0]

    if attack_type == 'main_class_anchor':
        # 强制对比主类字母，后面完全一样 (A 123 < Z 123)
        let_a, let_b = random.sample(string.ascii_uppercase, 2)
        if let_a > let_b: let_a, let_b = let_b, let_a
        A.cls_letters = let_a
        B.cls_letters = let_b
    elif attack_type == 'cutter_letter':
        let_a, let_b = random.sample(string.ascii_uppercase, 2)
        if let_a > let_b: let_a, let_b = let_b, let_a
        A.cutter1_let = let_a
        B.cutter1_let = let_b
    elif attack_type == 'cutter_digit_len':
        base = str(random.randint(1, 9))
        A.cutter1_num = base
        B.cutter1_num = base + str(random.randint(1, 9))
    elif attack_type == 'cutter_digit_val':
        # 【核心修复】构造严格的 A < B 小数陷阱 (例如 645 < 65)
        # 即使 A 看起来数字更长，但在分歧位 (第二位) A(4) < B(5)，所以 A < B
        first = str(random.randint(1, 9))
        val_second = random.randint(0, 8) 
        A.cutter1_num = first + str(val_second) + str(random.randint(1, 9)) # e.g. "645"
        B.cutter1_num = first + str(val_second + 1)                         # e.g. "65"
        # 验证逻辑：0.645 < 0.65，符合 A < B 的契约
    elif attack_type == 'year_near':
        base_year = random.randint(1950, 2020)
        A.has_year = B.has_year = True
        A.year = base_year
        B.year = base_year + random.randint(1, 10)
    elif attack_type == 'decimal_diff':
        # A < B，A 拿长的 (.45)，B 拿短的 (.5) -> 修复确认
        A.has_cls_decimal = B.has_cls_decimal = True
        A.cls_decimal = str(random.randint(10, 99))
        B.cls_decimal = str(random.randint(1, 9))
    elif attack_type == 'cutter2_diff':
        # A < B，A 用长数字 (.S29)，B 用短数字 (.S3) -> 修复确认
        let = random.choice(string.ascii_uppercase)
        A.cutter2_let = B.cutter2_let = let
        A.has_cutter2 = B.has_cutter2 = True
        A.cutter2_num = str(random.randint(10, 99))
        B.cutter2_num = str(random.randint(1, 9))
    elif attack_type == 'volume_diff':
        A.has_suffix = B.has_suffix = True
        A.suffix_type = 'v.'
        B.suffix_type = 'v.'
        base_vol = random.randint(1, 50)
        A.suffix_num = base_vol
        B.suffix_num = base_vol + 1
    elif attack_type == 'minimal_diff':
        # 【真正普适版】随机化差异位和具体数值，但严格保证 A < B
        minimal_type = random.choice(['cutter_last_digit', 'decimal_last_digit', 'year_suffix'])

        if minimal_type == 'cutter_last_digit':
            # 1. 随机生成前缀 (长度也可变)
            base = str(random.randint(1, 99)) 
            # 2. 从 0-9 中随机选两个不相等的数字并排序
            d1, d2 = sorted(random.sample(range(10), 2))
            A.cutter1_num = base + str(d1) # e.g. "643"
            B.cutter1_num = base + str(d2) # e.g. "648"
            
        elif minimal_type == 'decimal_last_digit':
            base = str(random.randint(1, 99))
            A.has_cls_decimal = B.has_cls_decimal = True
            # 同样利用排序保证方向
            d1, d2 = sorted(random.sample(range(10), 2))
            A.cls_decimal = base + str(d1)
            B.cls_decimal = base + str(d2)

        elif minimal_type == 'year_suffix':
            A.has_year = B.has_year = True
            base_year = random.randint(1900, 2025)
            A.year = B.year = base_year
            # 从可能的后缀字母表中随机选两个并排序
            s_pool = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'z']
            s1, s2 = sorted(random.sample(s_pool, 2))
            A.year_suffix = s1 # e.g. "c"
            B.year_suffix = s2 # e.g. "f"

    return A, B

def apply_custom_noise(text, intensity=1.0):
    """V3.3：可控强度 + 物理环境模拟噪音"""
    if intensity == 0: return text

    # 1. 结构噪音 (馆藏前缀)
    if random.random() < 0.15 * intensity:
        prefix = random.choice(PREFIXES)
        text = f"{prefix}\n{text}"
        
    # 2. 乱码注入 (模拟旁边书籍信息或出版社名被误扣取)
    if random.random() < 0.05 * intensity:
        garbage = random.choice(GARBAGE_TEXTS)
        if random.random() > 0.5:
            text = f"{garbage}\n{text}"
        else:
            text = f"{text}\n{garbage}"

    # 3. 字符混淆 (OCR 错误)
    chars = list(text)
    for i, char in enumerate(chars):
        if char in OCR_CONFUSION and random.random() < 0.005 * intensity:
            chars[i] = random.choice(OCR_CONFUSION[char])
    text = "".join(chars)
    
    # 4. YOLO 截断模拟 (丢弃最底下一行，模拟边框偏小)
    lines = text.split('\n')
    if len(lines) >= 3 and random.random() < 0.05 * intensity:
        lines = lines[:-1]

    # 5. 随机换行/空格变异
    joiner = random.choice(["\n", " "]) if random.random() < 0.2 else "\n"
    return joiner.join(lines)

# ==========================================
# 数据集生成逻辑
# ==========================================
def generate_dataset(num_samples=NUM_SAMPLES, output_file=OUTPUT_FILE):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"🚀 正在生成 Lost Book Robot V3.3 核心修复数据集 ({num_samples} 条)...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text_a', 'text_b', 'label'])

        for i in range(num_samples):
            dice = random.random()
            
            if dice < 0.80: # 排序类 (80%：Label 0 或 2)
                label = 0 if random.random() > 0.5 else 2
                
                if random.random() < 0.5:
                    lcc_a, lcc_b = generate_v33_hard_pair()
                    noise_lvl = 0 if random.random() < 0.2 else 1.0 # 保留 20% 纯净硬样本
                else:
                    lcc_a = LCCCallNumber()
                    lcc_b = copy.deepcopy(lcc_a)
                    lcc_b.cls_number += random.randint(1, 100)
                    noise_lvl = 1.0
                
                if label == 2: lcc_a, lcc_b = lcc_b, lcc_a
                
                text_a = apply_custom_noise(lcc_a.to_string(), noise_lvl)
                text_b = apply_custom_noise(lcc_b.to_string(), noise_lvl)

            else: # 重复类 (20%：Label 1) 专项修复区
                lcc = LCCCallNumber()
                label = 1
                lcc.has_year = True
                lcc.year = random.randint(2000, 2024)
                base_str = lcc.to_string()

                # 将 Label 1 分解为四个专项训练任务
                dup_strategy = random.choices(
                    ['prefix_diff', 'single_ocr_diff', 'standard_noise', 'clean'], 
                    weights=[0.3, 0.3, 0.3, 0.1]
                )[0]

                if dup_strategy == 'prefix_diff':
                    # 只有前缀不同，无其他噪音 (解决 Test 8 崩溃)
                    p1, p2 = random.sample(PREFIXES, 2)
                    text_a = f"{p1}\n{base_str}"
                    text_b = f"{p2}\n{base_str}"
                
                elif dup_strategy == 'single_ocr_diff':
                    # 只有单个 OCR 错误，前缀排版等全部一致 (解决 Test 9 崩溃)
                    text_a = base_str
                    chars_b = list(base_str)
                    possible_idx = [idx for idx, c in enumerate(chars_b) if c in OCR_CONFUSION]
                    if possible_idx:
                        replace_idx = random.choice(possible_idx)
                        chars_b[replace_idx] = random.choice(OCR_CONFUSION[chars_b[replace_idx]])
                    text_b = "".join(chars_b)
                
                elif dup_strategy == 'standard_noise':
                    # 模拟一本书的两张扫描件，各有各的随机噪音
                    text_a = apply_custom_noise(base_str, 0.8)
                    text_b = apply_custom_noise(base_str, 0.8)
                
                else: # 'clean'
                    # 作为锚点的完美对照组
                    text_a = base_str
                    text_b = base_str

            writer.writerow([text_a, text_b, label])
            if (i + 1) % (num_samples // 10 if num_samples > 10 else 1) == 0: 
                print(f"  进度: {i + 1}/{num_samples}")

    print(f"✅ V3.3 数据集生成完毕，准备拯救 Bi-LSTM 的三观！")

if __name__ == "__main__":
    random.seed(42)
    generate_dataset()