import random
import string
import csv
import os

# ==========================================
# 配置与常量
# ==========================================
OUTPUT_FILE = "data/synthetic_pairs/lcc_training_data.csv"
NUM_SAMPLES = 300000 

# 1. 真实场景前缀与变异字典 (模拟 OCR 对馆藏地的误读)
PREFIXES = ["OLIN", "URIS", "KROCH", "MANN", "LAW", "MATH", "FINE ARTS"]
PREFIX_MUTATIONS = {
    "OLIN": ["QLIN", "DLIN", "0LIN", "OLI", "OLIN "],
    "URIS": ["UR1S", "UR IS", "JRIS"],
    "BV": ["3V", "B V", "8V"],
    "PR": ["P R", "PIR", "P2"],
    "PQ": ["P Q", "Pa", "PO"]
}

# 2. 复杂的多语言与特殊版式后缀
SUFFIX_TYPES = [
    'v.', 'no.', 'c.', 'copy', 'vol.', 
    'Bd.', 'T.', 'Heft', 'suppl.', 'pt.', 
    '+', '++' # 超大尺寸书本标识
]

# 3. 真实的“垃圾文本”噪音库 (从 OCR 结果提取)
GARBAGE_TEXTS = [
    "LeoS. Olschki", "JOHN DONALD", "EKAOEEIT ANOE 42819196 473616424",
    "SH", "ISGH", "CSIC", "PEETERS", "ISBN 0-8298-0944-9", "CONHEL UNIVERSN GRY"
]

# 4. 基础字符混淆集 (降低触发概率，因为不是主要错误源)
OCR_CONFUSION = {
    '0': ['O', 'D', 'Q', 'C'], '1': ['I', 'l', '|', ']', '['],
    '2': ['Z', '7', '?'], '3': ['E', 'B', '8'], '4': ['A', 'H'],
    '5': ['S', '6'], '6': ['G', 'b', '5'], '8': ['B', '3', 'S'],
    '9': ['g', 'q'], 'B': ['8', 'E', '3'], 'D': ['0', 'O', 'Q'],
    'Z': ['2', '7'], '.': [',', ' ', "-", "'"]
}

# ==========================================
# 核心类：LCC 号码生成器
# ==========================================
class LCCCallNumber:
    def __init__(self):
        self.cls_letters = self._gen_letters()
        self.cls_number = random.randint(1, 9999)
        self.has_cls_decimal = random.random() > 0.6
        self.cls_decimal = random.randint(1, 999) if self.has_cls_decimal else None
        
        self.cutter1_let = random.choice(string.ascii_uppercase)
        self.cutter1_num = self._gen_cutter_num()
        self.cutter1_suffix = random.choice(['x', 'z']) if random.random() < 0.1 else ''
        
        self.has_cutter2 = random.random() > 0.3
        self.cutter2_let = random.choice(string.ascii_uppercase)
        self.cutter2_num = self._gen_cutter_num()
        self.cutter2_suffix = random.choice(['x', 'z']) if random.random() < 0.1 else ''
        
        self.has_year = random.random() > 0.4
        self.year = random.randint(1850, 2024)
        self.year_suffix = random.choice(['a', 'b', 'c', 'x', 'z']) if random.random() < 0.1 else ''
        
        self.has_suffix = random.random() > 0.6 # 提高后缀出现率
        self.suffix_type = random.choice(SUFFIX_TYPES)
        self.suffix_num = random.randint(1, 100)

    def _gen_letters(self):
        length = random.choices([1, 2, 3], weights=[0.2, 0.7, 0.1])[0]
        return ''.join(random.choices(string.ascii_uppercase, k=length))

    def _gen_cutter_num(self):
        return str(random.randint(2, 9999))

    def to_string(self):
        parts = []
        
        # Class
        cls_str = f"{self.cls_letters} {self.cls_number}" # 默认带个空格
        if self.has_cls_decimal:
            cls_str += f".{self.cls_decimal}"
        parts.append(cls_str)
        
        # Cutters & Year
        parts.append(f".{self.cutter1_let}{self.cutter1_num}{self.cutter1_suffix}")
        if self.has_cutter2:
            parts.append(f"{self.cutter2_let}{self.cutter2_num}{self.cutter2_suffix}")
        if self.has_year:
            parts.append(f"{self.year}{self.year_suffix}")
            
        # 复杂后缀处理 (e.g., Bd.33, V. 4, ++)
        if self.has_suffix:
            if self.suffix_type in ['+', '++']:
                parts.append(self.suffix_type)
            else:
                sep = " " if random.random() > 0.5 else ""
                parts.append(f"{self.suffix_type}{sep}{self.suffix_num}")

        return " \n ".join(parts) # 默认用 \n 或空格连接，后续交由 noise 函数打乱

    def clone_and_increment(self):
        """生成一个在排序上严格大于当前对象的 LCC 号码"""
        new_lcc = LCCCallNumber()
        # 深度复制 (省略部分重复代码，沿用你之前的克隆逻辑)
        new_lcc.__dict__ = self.__dict__.copy()

        level = random.choices(
            ['cls_let', 'cls_num', 'cut1', 'cut2', 'year', 'year_suffix', 'suffix', 'add_structure'],
            weights=[0.05, 0.1, 0.25, 0.2, 0.15, 0.1, 0.1, 0.05]
        )[0]

        if level == 'cls_let':
            if not self.cls_letters.endswith('Z'):
                new_lcc.cls_letters = self.cls_letters[:-1] + chr(ord(self.cls_letters[-1]) + 1)
            else:
                new_lcc.cls_number += 1
        elif level == 'cls_num':
            new_lcc.cls_number += random.randint(1, 5)
        elif level == 'cut1':
            if not self.cutter1_num.endswith('9'):
                new_lcc.cutter1_num = str(int(self.cutter1_num) + 1)
            else:
                new_lcc.cutter1_num += str(random.randint(1, 9))
        elif level == 'cut2':
            if not self.has_cutter2:
                new_lcc.has_cutter2 = True
            else:
                if not self.cutter2_num.endswith('9'):
                    new_lcc.cutter2_num = str(int(self.cutter2_num) + 1)
                else:
                    new_lcc.cutter2_num += str(random.randint(1, 9))
        elif level == 'year':
            if not self.has_year:
                new_lcc.has_year = True
            else:
                new_lcc.year += random.randint(1, 3)
        elif level == 'year_suffix':
            if not self.has_year:
                new_lcc.has_year = True
            if self.year_suffix == '': new_lcc.year_suffix = 'a'
            elif self.year_suffix == 'z':
                new_lcc.year += 1
                new_lcc.year_suffix = ''
            else:
                new_lcc.year_suffix = chr(ord(self.year_suffix) + 1)
        elif level == 'suffix':
            if not self.has_suffix:
                new_lcc.has_suffix = True
            else:
                new_lcc.suffix_num += 1
        elif level == 'add_structure':
             if not self.has_cutter2: new_lcc.has_cutter2 = True
             elif not self.has_year: new_lcc.has_year = True
             elif not self.has_suffix: new_lcc.has_suffix = True
             else: new_lcc.year += 1
        return new_lcc

# ==========================================
# 核心噪音引擎
# ==========================================
def apply_real_world_ocr_noise(text):
    """注入真实的 OCR 结构性噪音"""
    
    # 1. 结构性粘连与断裂 (Space Jittering)
    parts = text.split()
    noisy_parts = []
    for part in parts:
        # 变异知名大类 (如 BV -> 3V, PR -> P R)
        for key, mutations in PREFIX_MUTATIONS.items():
            if key in part and random.random() < 0.2:
                part = part.replace(key, random.choice(mutations))
        
        # 随机拆分 (例如 330 -> 3 30)
        if random.random() < 0.05 and len(part) > 2:
            split_idx = random.randint(1, len(part)-1)
            part = part[:split_idx] + " " + part[split_idx:]
            
        noisy_parts.append(part)

    # 随机粘连 (例如 .I8 A78 -> .I8A78)
    join_char = "" if random.random() < 0.15 else " "
    if random.random() < 0.3:
        join_char = "\n"  # 模拟换行
    
    result = join_char.join(noisy_parts)

    # 2. 单字符混淆 (Typo Injection)
    chars = list(result)
    for i, char in enumerate(chars):
        if char in OCR_CONFUSION and random.random() < 0.01:
            chars[i] = random.choice(OCR_CONFUSION[char])
    result = "".join(chars)

    # 3. 前缀与垃圾文本注入
    if random.random() < 0.2:  # 20% 概率加前缀
        prefix = random.choice(PREFIXES)
        if prefix in PREFIX_MUTATIONS and random.random() < 0.3:
            prefix = random.choice(PREFIX_MUTATIONS[prefix])
        result = f"{prefix} {result}"
        
    if random.random() < 0.05:  # 5% 概率混入周围的垃圾文本
        garbage = random.choice(GARBAGE_TEXTS)
        if random.random() > 0.5:
            result = f"{garbage}\n{result}"
        else:
            result = f"{result}\n{garbage}"

    return result

# ==========================================
# 数据集生成逻辑
# ==========================================
def generate_dataset(num_samples=NUM_SAMPLES, output_file=OUTPUT_FILE):
    print(f"Generating {num_samples} robust samples...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text_a', 'text_b', 'label'])

        for i in range(num_samples):
            # 调整了类别分布：真实书架上，0 (A<B) 占绝大多数，1 (重复) 较少，2 (错位) 偶发
            data_type = random.choices([0, 1, 2], weights=[0.60, 0.15, 0.25])[0]

            if data_type == 0:
                lcc_a = LCCCallNumber()
                lcc_b = lcc_a.clone_and_increment()
                clean_a, clean_b = lcc_a.to_string(), lcc_b.to_string()
                label = 0
            elif data_type == 1:
                lcc = LCCCallNumber()
                clean_a = lcc.to_string()
                clean_b = clean_a  # 底层真值相同
                label = 1
            else:
                lcc_b = LCCCallNumber()
                lcc_a = lcc_b.clone_and_increment()
                clean_a, clean_b = lcc_a.to_string(), lcc_b.to_string()
                label = 2

            # 核心改进：即使是 A==B，我们也会分别独立施加噪音！
            # 这迫使 LSTM 学习到 "QLIN PR 123" 和 "OLIN P R 123" 在语义上是相等的。
            text_a = apply_real_world_ocr_noise(clean_a)
            text_b = apply_real_world_ocr_noise(clean_b)

            writer.writerow([text_a, text_b, label])

            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i + 1}/{num_samples} samples generated")

    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    random.seed(42)
    generate_dataset(1000) # 测试跑 1000 条