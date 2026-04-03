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
GARBAGE_TEXTS = ["LeoS. Olschki", "JOHN DONALD", "EKAOEEIT ANOE", "CSIC", "PEETERS", "ISBN 0-8298-0944-9"]
GARBAGE_PREFIX = ["VOL", "COPY", "MAIN", "ANNEX", "BOOK", "SHELF"]

OCR_CONFUSION = {
    '0': ['O', 'Q', 'D'],
    '1': ['I', 'l', '|', ']', '(', ')'],
    '2': ['Z', 'z'],
    '5': ['S', 's'],
    '6': ['G', 'b'],
    '8': ['B', '&'],
    '9': ['g', 'q'],
    'B': ['8'],
    'D': ['0', 'O'],
    'Z': ['2', 'z'],
    '.': [',', '-', " ", "'", "`"],
    'v': ['V', 'Y'],
    'O': ['0', 'o', 'Q']
}

class LCCCallNumber:
    def __init__(self):
        self.cls_letters = self._gen_letters()
        self.cls_number = random.randint(1, 9999)
        self.has_cls_decimal = random.random() > 0.4
        self.cls_decimal = str(random.randint(1, 999)) if self.has_cls_decimal else ""
        
        self.cutter1_let = random.choice(string.ascii_uppercase)
        self.cutter1_num = str(random.randint(2, 999))
        # V7: 基础生成也放开到所有大小写字母
        self.cutter1_workmark = random.choice(string.ascii_letters) if random.random() < 0.15 else ""
        
        self.has_cutter2 = random.random() > 0.4
        self.cutter2_let = random.choice(string.ascii_uppercase) if self.has_cutter2 else ""
        self.cutter2_num = str(random.randint(2, 999)) if self.has_cutter2 else ""
        self.cutter2_workmark = random.choice(string.ascii_letters) if self.has_cutter2 and random.random() < 0.1 else ""
        
        self.has_year = random.random() > 0.5
        self.year = random.randint(1880, 2025) if self.has_year else 0
        self.year_suffix = random.choice(string.ascii_letters) if self.has_year and random.random() < 0.1 else ""
        
        self.has_suffix = random.random() > 0.8
        suffix_pool = ['v.', 'V.', 'no.', 'c.', 'copy', 'vol.', 'Bd.', 'Heft', 'suppl.', 'pt.']
        self.suffix_type = random.choice(suffix_pool) if self.has_suffix else ""
        self.suffix_num = random.randint(1, 50) if self.has_suffix else 0
        
        self.oversize_mark = random.choice(['+', '++']) if random.random() < 0.05 else ""

    def _gen_letters(self):
        length = random.choices([1, 2, 3], weights=[0.2, 0.7, 0.1])[0]
        return ''.join(random.choices(string.ascii_uppercase, k=length))

    def to_string(self):
        parts = []
        cls_str = f"{self.cls_letters} {self.cls_number}"
        if self.has_cls_decimal: 
            space_before = " " if random.random() < 0.3 else ""
            cls_dot = "." if random.random() > 0.2 else " "
            cls_str += f"{space_before}{cls_dot}{self.cls_decimal}"
        parts.append(cls_str)
        
        # 让 Cutter 前面的空格和点号呈现 50% 的极度随机状态
        space1 = " " if random.random() < 0.5 else ""
        dot1 = "." if random.random() < 0.5 else "" 
        
        # 偶尔制造 "隐式 Cutter" 现象 (连点号和空格都没了)
        if random.random() < 0.1:
            space1, dot1 = "", ""
            
        parts.append(f"{space1}{dot1}{self.cutter1_let}{self.cutter1_num}{self.cutter1_workmark}")
        
        if self.has_cutter2: 
            space2 = " " if random.random() < 0.3 else ""
            dot2 = "." if random.random() > 0.5 else " "
            parts.append(f"{space2}{dot2}{self.cutter2_let}{self.cutter2_num}{self.cutter2_workmark}")
            
        if self.has_year: 
            parts.append(f"{self.year}{self.year_suffix}")
            
        if self.has_suffix:
            st = self.suffix_type
            # 随机突变卷册前缀的大小写 (打破 In_Order 样本的绝对对称)
            dice = random.random()
            if dice < 0.33:
                st = st.upper()       # v. -> V.
            elif dice < 0.66:
                st = st.lower()       # V. -> v.
            else:
                st = st.capitalize()  # vol. -> Vol.
                
            parts.append(f"{st}{self.suffix_num}")
            
        if self.oversize_mark:
            parts.append(self.oversize_mark)
            
        return " ".join(parts)


# ==========================================
# V7 层级分化生成器
# ==========================================
def _sync_tail_structure(A: LCCCallNumber, B: LCCCallNumber):
    B.has_year = A.has_year
    B.has_suffix = A.has_suffix
    B.has_cutter2 = A.has_cutter2
    B.has_cls_decimal = A.has_cls_decimal

def _randomize_tail_structure(A: LCCCallNumber, B: LCCCallNumber):
    B.has_year = random.random() > 0.5
    B.has_suffix = random.random() > 0.8
    B.has_cutter2 = random.random() > 0.4

def generate_v7_hierarchical_logic():
    """
    V7 细粒度全节点生成器
    加入全字母表支持，并随机混合大小写，迫使模型学习大小写无关的字典序
    """
    A = LCCCallNumber()
    B = LCCCallNumber()

    levels = [
        'class_let', 'class_num', 'decimal',
        'c1_let', 'c1_num', 'c1_wm',
        'c2_let', 'c2_num', 'c2_wm',
        'year', 'year_suf', 'vol'
    ]

    weights = [1, 1, 6, 1, 1, 8, 1, 1, 8, 2, 2, 8]
    diverge_level = random.choices(levels, weights=weights, k=1)[0]
    idx = levels.index(diverge_level)

    # 步骤 1：分歧点之前保持一致
    if idx > 0: B.cls_letters = A.cls_letters
    if idx > 1: B.cls_number = A.cls_number
    if idx > 2:
        B.has_cls_decimal = A.has_cls_decimal
        B.cls_decimal = A.cls_decimal
    if idx > 3: B.cutter1_let = A.cutter1_let
    if idx > 4: B.cutter1_num = A.cutter1_num
    if idx > 5: B.cutter1_workmark = A.cutter1_workmark
    if idx > 6:
        A.has_cutter2 = B.has_cutter2 = True
        B.cutter2_let = A.cutter2_let
    if idx > 7: B.cutter2_num = A.cutter2_num
    if idx > 8: B.cutter2_workmark = A.cutter2_workmark
    if idx > 9:
        A.has_year = B.has_year = True
        B.year = A.year
    if idx > 10: B.year_suffix = A.year_suffix

    # 步骤 2：在分歧点制造 A < B
    if diverge_level == 'class_let':
        l1, l2 = sorted(random.sample(string.ascii_uppercase, 2))
        base = "".join(random.choices(string.ascii_uppercase, k=random.choice([0,1])))
        A.cls_letters = base + l1
        B.cls_letters = base + l2
        if random.random() < 0.5:
            _randomize_tail_structure(A, B)
        else:
            _sync_tail_structure(A, B)

    elif diverge_level == 'class_num':
        B.cls_number = A.cls_number + random.randint(1, 500)
        if random.random() < 0.5:
            _randomize_tail_structure(A, B)
        else:
            _sync_tail_structure(A, B)

    elif diverge_level == 'decimal':
        A.has_cls_decimal = B.has_cls_decimal = True
        if random.random() < 0.5:
            A.cls_decimal = str(random.randint(1, 9))
            B.cls_decimal = A.cls_decimal + str(random.randint(1, 9))
        else:
            base = random.randint(1, 8)
            A.cls_decimal = str(base) + str(random.randint(1, 9))
            B.cls_decimal = str(base + 1)
        if random.random() < 0.5:
            _randomize_tail_structure(A, B)
        else:
            _sync_tail_structure(A, B)

    elif diverge_level == 'c1_let':
        l1, l2 = sorted(random.sample(string.ascii_uppercase, 2))
        A.cutter1_let = l1; B.cutter1_let = l2
        _sync_tail_structure(A, B)

    elif diverge_level == 'c1_num':
        if random.random() < 0.5: 
            A.cutter1_num = str(random.randint(1, 9))
            B.cutter1_num = A.cutter1_num + str(random.randint(1, 9))
        else: 
            base = random.randint(1, 8)
            A.cutter1_num = str(base) + str(random.randint(1, 9))
            B.cutter1_num = str(base + 1)
        _sync_tail_structure(A, B)

    elif diverge_level == 'c1_wm':
        # V7: 保证字典序正确的前提下，随机赋予大小写
        c1, c2 = sorted(random.sample(string.ascii_lowercase, 2))
        A.cutter1_workmark = c1.upper() if random.random() < 0.5 else c1
        B.cutter1_workmark = c2.upper() if random.random() < 0.5 else c2
        _sync_tail_structure(A, B)

    elif diverge_level == 'c2_let':
        A.has_cutter2 = B.has_cutter2 = True
        l1, l2 = sorted(random.sample(string.ascii_uppercase, 2))
        A.cutter2_let = l1; B.cutter2_let = l2
        _sync_tail_structure(A, B)

    elif diverge_level == 'c2_num':
        A.has_cutter2 = B.has_cutter2 = True
        if random.random() < 0.5:
            A.cutter2_num = str(random.randint(1, 9))
            B.cutter2_num = A.cutter2_num + str(random.randint(1, 9))
        else:
            base = random.randint(1, 8)
            A.cutter2_num = str(base) + str(random.randint(1, 9))
            B.cutter2_num = str(base + 1)
        _sync_tail_structure(A, B)

    elif diverge_level == 'c2_wm':
        A.has_cutter2 = B.has_cutter2 = True
        c1, c2 = sorted(random.sample(string.ascii_lowercase, 2))
        A.cutter2_workmark = c1.upper() if random.random() < 0.5 else c1
        B.cutter2_workmark = c2.upper() if random.random() < 0.5 else c2
        _sync_tail_structure(A, B)

    elif diverge_level == 'year':
        A.has_year = B.has_year = True
        B.year = A.year + random.randint(1, 10)
        _sync_tail_structure(A, B)

    elif diverge_level == 'year_suf':
        A.has_year = B.has_year = True
        c1, c2 = sorted(random.sample(string.ascii_lowercase, 2))
        A.year_suffix = c1.upper() if random.random() < 0.5 else c1
        B.year_suffix = c2.upper() if random.random() < 0.5 else c2
        _sync_tail_structure(A, B)

    elif diverge_level == 'vol':
        A.has_suffix = B.has_suffix = True
        A.suffix_type = B.suffix_type = random.choice(['v.', 'pts', 'no.'])
        B.suffix_num = A.suffix_num + random.randint(1, 5)
        _sync_tail_structure(A, B)

    return A, B


def apply_custom_noise(text, intensity=1.0):
    if intensity == 0: return text

    # Prefix Hard Negatives: 10% 概率在开头塞入和索书号无关的干扰词
    if random.random() < 0.1:
        text = f"{random.choice(GARBAGE_PREFIX)} {text}"

    if random.random() < 0.05 * intensity:
        garbage = random.choice(GARBAGE_TEXTS)
        text = f"{text} {garbage}" if random.random() > 0.5 else f"{garbage} {text}"

    chars = list(text)
    new_chars = []

    for i, char in enumerate(chars):
        if char in OCR_CONFUSION and random.random() < 0.005 * intensity:
            char = random.choice(OCR_CONFUSION[char])

        new_chars.append(char)

        if char != " " and random.random() < 0.05 * intensity:
            new_chars.append(" ")

    text_with_spaces = "".join(new_chars)

    if random.random() < 0.15 * intensity:
        parts = text_with_spaces.split(" ")
        if len(parts) > 1:
            idx = random.randint(0, len(parts) - 2)
            parts[idx] = parts[idx] + parts[idx+1]
            del parts[idx+1]
        text_with_spaces = " ".join(parts)

    return text_with_spaces.strip()


def apply_spacing_noise_only(text):
    """
    V7 终极脱敏版：专用于 Duplicate 样本
    保留排版断裂的同时，加入垃圾干扰词和前缀截断，培养模型的"垃圾免疫力"
    """
    # 1. 垃圾干扰词免疫 (5%概率)
    if random.random() < 0.05:
        garbage = random.choice(GARBAGE_TEXTS)
        text = f"{text} {garbage}" if random.random() > 0.5 else f"{garbage} {text}"
        
    # 2. 模拟真实 OCR 的前缀截断 (例如 OLIN 识别成了 OLI)
    for prefix in PREFIXES:
        if f"{prefix} " in text and random.random() < 0.05:
            truncated = prefix[:-1] # 砍掉最后一个字母
            text = text.replace(f"{prefix} ", f"{truncated} ")
            break # 每次只破坏一个前缀

    chars = list(text)
    new_chars = []

    for char in chars:
        new_chars.append(char)
        # 3. 空格撕裂
        if char != " " and random.random() < 0.05:
            new_chars.append(" ")

    text_with_spaces = "".join(new_chars)

    # 4. 空格粘连
    if random.random() < 0.15:
        parts = text_with_spaces.split(" ")
        if len(parts) > 1:
            idx = random.randint(0, len(parts) - 2)
            parts[idx] = parts[idx] + parts[idx+1]
            del parts[idx+1]
        text_with_spaces = " ".join(parts)

    return text_with_spaces.strip()


def generate_dataset(num_samples=NUM_SAMPLES, output_file=OUTPUT_FILE):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"🚀 开始生成 V7 终极大满贯脱敏版数据集...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text_a', 'text_b', 'label'])

        for i in range(num_samples):
            dice = random.random()

            # 前缀完全独立生成，彻底堵死"前缀不对称=Duplicate"的捷径
            # 30% 概率无前缀，70% 概率从池子里随机抓
            prefix_a = f"{random.choice(PREFIXES)} " if random.random() < 0.7 else ""
            prefix_b = f"{random.choice(PREFIXES)} " if random.random() < 0.7 else ""

            if dice < 0.80:
                # 生成 0(In_Order) 和 2(Out_of_Order)
                label = 0 if random.random() > 0.5 else 2

                # 【新增】强行制造 20% 的 "微小差异困难样本" (Micro-diff Hard Negatives)
                is_micro_diff = random.random() < 0.2

                lcc_a, lcc_b = generate_v7_hierarchical_logic()

                if is_micro_diff:
                    # 强行让 A 和 B 在开头和中间完全一致，只在末尾差异
                    lcc_b = copy.deepcopy(lcc_a)
                    # 只在卷册或年份上做 +1 操作
                    if random.random() < 0.5:
                        lcc_a.has_suffix = lcc_b.has_suffix = True
                        lcc_a.suffix_type = lcc_b.suffix_type = 'v.'
                        lcc_b.suffix_num = lcc_a.suffix_num + 1
                    else:
                        lcc_a.has_year = lcc_b.has_year = True
                        lcc_b.year = lcc_a.year + 1

                if label == 2:
                    lcc_a, lcc_b = lcc_b, lcc_a

                # 如果是微小差异样本，降低噪音干扰，让模型看清楚那个不同的数字
                intensity = 0.0 if is_micro_diff else 1.0
                text_a = apply_custom_noise(prefix_a + lcc_a.to_string(), intensity=intensity)
                text_b = apply_custom_noise(prefix_b + lcc_b.to_string(), intensity=intensity)

            else:
                # 生成 1(Duplicate)
                label = 1
                lcc_a = LCCCallNumber()
                lcc_b = copy.deepcopy(lcc_a)

                # 增加 30% 的概率让前缀完全一样，防止模型把"前缀不同"当成 Duplicate 的唯一特征
                if random.random() < 0.3:
                    p = f"{random.choice(PREFIXES)} " if random.random() < 0.5 else ""
                    prefix_a = prefix_b = p

                text_a_base = prefix_a + lcc_a.to_string()
                text_b_base = prefix_b + lcc_b.to_string()

                text_a = apply_spacing_noise_only(text_a_base)
                text_b = apply_spacing_noise_only(text_b_base)

            writer.writerow([text_a, text_b, label])

            if (i+1) % 50000 == 0:
                print(f"  进度: {i+1}/{num_samples}")

    print(f"✅ V7 数据集生成完毕！请重置模型权重，重新进行训练。")

if __name__ == "__main__":
    generate_dataset()