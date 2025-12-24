import random
import string
import csv
import os

# ==========================================
# 配置与常量
# ==========================================

OUTPUT_FILE = "lcc_training_data.csv"
NUM_SAMPLES = 300000 

OCR_CONFUSION = {
    '0': ['O', 'o', 'D', 'Q'],
    '1': ['I', 'l', '|', '!', 'i'],
    '2': ['Z', 'z'],
    '3': ['E'],
    '4': ['A'],
    '5': ['S', 's'],
    '6': ['G', 'b'],
    '8': ['B', '3'],
    'B': ['8', 'E'],
    'D': ['0', 'O'],
    'Z': ['2'],
    '.': [',', ' ', '', '-'], 
    ' ': ['|', '\n', '_', '']  
}

PREFIXES = ["OLIN", "REF", "KROCH", "URIS", "MANN", "LAW", "MATH", "FINE ARTS"]

# ==========================================
# 核心类：LCC 号码生成器
# ==========================================

class LCCCallNumber:
    def __init__(self):
        self.cls_letters = self._gen_letters()
        self.cls_number = random.randint(1, 9999)
        self.cutter1_let = random.choice(string.ascii_uppercase)
        self.cutter1_num = self._gen_cutter_num()
        # Cutter 2 (30% 概率存在)
        self.has_cutter2 = random.random() > 0.3
        self.cutter2_let = random.choice(string.ascii_uppercase)
        self.cutter2_num = self._gen_cutter_num()
        # Year (80% 概率存在)
        self.has_year = random.random() > 0.2
        self.year = random.randint(1900, 2024)

    def _gen_letters(self):
        length = random.choices([1, 2, 3], weights=[0.2, 0.7, 0.1])[0]
        return ''.join(random.choices(string.ascii_uppercase, k=length))

    def _gen_cutter_num(self):
        # 避免生成以0开头的数字字符串，LCC Cutter通常不写0
        return str(random.randint(2, 999))

    def to_string(self):
        s = f"{self.cls_letters}{self.cls_number} .{self.cutter1_let}{self.cutter1_num}"
        if self.has_cutter2:
            s += f" {self.cutter2_let}{self.cutter2_num}"
        if self.has_year:
            s += f" {self.year}"
        return s

    def clone_and_increment(self):
        new_lcc = LCCCallNumber()
        # 深度复制属性
        new_lcc.cls_letters = self.cls_letters
        new_lcc.cls_number = self.cls_number
        new_lcc.cutter1_let = self.cutter1_let
        new_lcc.cutter1_num = self.cutter1_num
        new_lcc.has_cutter2 = self.has_cutter2
        new_lcc.cutter2_let = self.cutter2_let
        new_lcc.cutter2_num = self.cutter2_num
        new_lcc.has_year = self.has_year
        new_lcc.year = self.year

        # 尝试变异 (最多尝试 10 次，防止死循环)
        for _ in range(10):
            # 新增策略：add_structure (例如原本没有 Cutter2，现在加上)
            level = random.choice(['class_let', 'class_num', 'cutter1', 'cutter2', 'year', 'add_structure'])

            if level == 'class_let':
                if self.cls_letters.endswith('Z'): continue
                last_char = self.cls_letters[-1]
                new_char = chr(ord(last_char) + 1)
                new_lcc.cls_letters = self.cls_letters[:-1] + new_char

            elif level == 'class_num':
                new_lcc.cls_number += random.randint(1, 100)

            elif level == 'cutter1':
                choice = random.choice(['letter', 'digit_val', 'digit_len'])
                if choice == 'letter':
                    if self.cutter1_let == 'Z': continue
                    new_lcc.cutter1_let = chr(ord(self.cutter1_let) + 1)
                elif choice == 'digit_val':
                    # 只有位数不变才能直接加值 (例: 12->13 OK, 9->10 NO)
                    if not self.cutter1_num.endswith('9'):
                        val = int(self.cutter1_num) + 1
                        if len(str(val)) == len(self.cutter1_num):
                            new_lcc.cutter1_num = str(val)
                        else:
                            new_lcc.cutter1_num += str(random.randint(1, 9))
                    else:
                        new_lcc.cutter1_num += str(random.randint(1, 9))
                elif choice == 'digit_len':
                    # 扩展长度 (0.5 -> 0.53) 永远更大
                    new_lcc.cutter1_num += str(random.randint(1, 9))

            elif level == 'cutter2':
                if not self.has_cutter2: continue
                # Cutter2 逻辑同 Cutter1
                if not self.cutter2_num.endswith('9'):
                    val = int(self.cutter2_num) + 1
                    if len(str(val)) == len(self.cutter2_num):
                        new_lcc.cutter2_num = str(val)
                    else:
                        new_lcc.cutter2_num += str(random.randint(1, 9))
                else:
                    new_lcc.cutter2_num += str(random.randint(1, 9))

            elif level == 'add_structure':
                # 【新增】: 如果A没有Cutter2，B加上Cutter2，则 B > A
                if not self.has_cutter2:
                    new_lcc.has_cutter2 = True
                    # 确保生成的 cutter2 是合理的
                    new_lcc.cutter2_let = random.choice(string.ascii_uppercase)
                    new_lcc.cutter2_num = str(random.randint(2, 99))
                # 或者如果A没有Year，B加上Year
                elif not self.has_year:
                    new_lcc.has_year = True
                    new_lcc.year = random.randint(2000, 2024)
                else:
                    continue

            elif level == 'year' and self.has_year:
                new_lcc.year += random.randint(1, 10)

            # 只要字符串变了，就认为变异成功
            if new_lcc.to_string() != self.to_string():
                return new_lcc

        # 如果10次都没成功（极罕见），强制增加年份或数字确保 B > A
        new_lcc.year += 1
        return new_lcc

# ==========================================
# 噪声注入模块
# ==========================================

def apply_noise(text):
    """将干净的 LCC 字符串转化为带有 OCR 噪声的版本

    噪声级别 (更接近真实场景):
    - 字符替换概率: 1% (极低，模拟高质量OCR)
    - 前缀添加概率: 3% (偶尔有图书馆前缀)
    - 杂质插入概率: 0.3% (几乎无杂质)
    - 格式清洗概率: 5% (少量格式问题)
    """
    chars = list(text)
    noisy_chars = []

    # 随机添加前缀 (低概率)
    if random.random() < 0.03:  # 降低到 3%
        prefix = random.choice(PREFIXES)
        sep = random.choice([' ', '|'])
        noisy_chars.extend(list(prefix + sep))

    for char in chars:
        # 字符替换 (极低概率 - 模拟高质量OCR)
        if char in OCR_CONFUSION and random.random() < 0.01:  # 降低到 1%
            noisy_chars.append(random.choice(OCR_CONFUSION[char]))
        else:
            noisy_chars.append(char)
        # 插入杂质 (极低概率)
        if random.random() < 0.003:  # 降低到 0.3%
            noisy_chars.append(random.choice(['.', ' ']))

    result = "".join(noisy_chars)

    # 格式清洗 (低概率)
    if random.random() < 0.05:  # 降低到 5%
        result = result.replace(" ", random.choice(["|", "  "]))

    return result

# ==========================================
# 主生成逻辑
# ==========================================

def generate_dataset():
    data = []
    print(f"Generating {NUM_SAMPLES} pairs...")
    
    for _ in range(NUM_SAMPLES):
        book_a = LCCCallNumber()
        
        # 0: In Order (A < B)
        # 1: Out of Order (B < A) -> 注意这里原本是 A>B，交换后变成 B<A
        # 2: Duplicate (A == B)
        label = random.choices([0, 1, 2], weights=[0.45, 0.45, 0.1])[0]
        
        str_a_clean = book_a.to_string()
        
        if label == 2:
            str_b_clean = str_a_clean
        else:
            # 生成 B > A
            book_b = book_a.clone_and_increment()
            str_b_clean = book_b.to_string()
        
        str_a_noisy = apply_noise(str_a_clean)
        str_b_noisy = apply_noise(str_b_clean)
        
        # 根据标签决定物理顺序
        if label == 1:
            # Label 1 代表 "顺序错误"。既然 B > A，我们放入 (B, A)，这样前者比后者大，顺序就是错的。
            final_a, final_b = str_b_noisy, str_a_noisy
        else:
            # Label 0 代表 "顺序正确"。放入 (A, B)。
            final_a, final_b = str_a_noisy, str_b_noisy
            
        data.append([final_a, final_b, label])
        
    # 保存 CSV
    # 1. 使用 utf-8-sig 让 Excel 打开不乱码
    # 2. quoting=csv.QUOTE_ALL 确保换行符被正确包裹，不会破坏 CSV 结构
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text_a", "text_b", "label"]) 
        writer.writerows(data)
        
    print(f"Done! Saved to {OUTPUT_FILE}")
    
    print("\n--- Sample Data Preview ---")
    for i in range(5):
        print(f"Sample {i}: Label {data[i][2]}")
        # 使用 repr() 打印以显式显示 \n
        print(f"  A: {data[i][0]!r}")
        print(f"  B: {data[i][1]!r}")

if __name__ == "__main__":
    generate_dataset()