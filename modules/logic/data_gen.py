import random
import string
import csv
import os

# ==========================================
# 配置与常量
# ==========================================

OUTPUT_FILE = "lcc_training_data.csv"
NUM_SAMPLES = 300000 

# [修复 1] OCR 混淆集：移除自身映射，确保噪声有效性
# 基于常见 OCR 错误（形近字、噪点）
OCR_CONFUSION = {
    '0': ['O', 'o', 'D', 'Q', 'C', '@'],
    '1': ['I', 'l', '|', '!', 'i', ']', '['],
    '2': ['Z', 'z', '7', '?'],
    '3': ['E', 'B', '8'],
    '4': ['A', 'H', 'h'],
    '5': ['S', 's', '6'],
    '6': ['G', 'b', '5'],
    '8': ['B', '3', 'S'],
    '9': ['g', 'q'],
    'B': ['8', 'E', '3'],
    'D': ['0', 'O', 'Q'],
    'Z': ['2', '7'],
    '.': [',', ' ', '-', "'"], 
    ' ': ['|', '_', '.', ','] 
}

PREFIXES = ["OLIN", "REF", "KROCH", "URIS", "MANN", "LAW", "MATH", "FINE ARTS"]

# ==========================================
# 核心类：LCC 号码生成器
# ==========================================

class LCCCallNumber:
    def __init__(self):
        self.cls_letters = self._gen_letters()
        self.cls_number = random.randint(1, 9999)
        self.has_cls_decimal = random.random() > 0.6
        self.cls_decimal = random.randint(1, 999) if self.has_cls_decimal else None
        
        # Cutter 1
        self.cutter1_let = random.choice(string.ascii_uppercase)
        self.cutter1_num = self._gen_cutter_num()
        self.cutter1_suffix = random.choice(['x', 'z']) if random.random() < 0.1 else ''
        
        # Cutter 2
        self.has_cutter2 = random.random() > 0.3
        self.cutter2_let = random.choice(string.ascii_uppercase)
        self.cutter2_num = self._gen_cutter_num()
        self.cutter2_suffix = random.choice(['x', 'z']) if random.random() < 0.1 else ''
        
        # Year
        self.has_year = random.random() > 0.4
        self.year = random.randint(1900, 2024)
        self.year_suffix = random.choice(['a', 'b', 'c', 'x', 'z']) if random.random() < 0.1 else ''
        
        # Suffix (Vol/Copy)
        self.has_suffix = random.random() > 0.9
        self.suffix_type = random.choice(['v.', 'no.', 'c.', 'copy', 'vol.'])
        self.suffix_num = random.randint(1, 50)

    def _gen_letters(self):
        length = random.choices([1, 2, 3], weights=[0.2, 0.7, 0.1])[0]
        return ''.join(random.choices(string.ascii_uppercase, k=length))

    def _gen_cutter_num(self):
        return str(random.randint(2, 9999)) # 避免 0 开头

    def to_string(self, layout="random"):
        if layout == "random":
            layout = random.choices(['horizontal', 'vertical'], weights=[0.2, 0.8])[0]

        parts = []
        
        # Class
        cls_str = f"{self.cls_letters}{self.cls_number}"
        if self.has_cls_decimal:
            cls_str += f".{self.cls_decimal}"
        parts.append(cls_str)
        
        # Cutter 1
        parts.append(f".{self.cutter1_let}{self.cutter1_num}{self.cutter1_suffix}")
        
        # Cutter 2
        if self.has_cutter2:
            parts.append(f"{self.cutter2_let}{self.cutter2_num}{self.cutter2_suffix}")
            
        # Year
        if self.has_year:
            parts.append(f"{self.year}{self.year_suffix}")
            
        # Suffix
        if self.has_suffix:
            parts.append(f"{self.suffix_type}{self.suffix_num}")

        if layout == 'vertical':
            return "\n".join(parts)
        else:
            return " ".join(parts)

    def clone_and_increment(self):
        """生成一个在排序上严格大于当前对象的 LCC 号码"""
        new_lcc = LCCCallNumber()
        # 深度复制
        new_lcc.cls_letters = self.cls_letters
        new_lcc.cls_number = self.cls_number
        new_lcc.has_cls_decimal = self.has_cls_decimal
        new_lcc.cls_decimal = self.cls_decimal
        new_lcc.cutter1_let = self.cutter1_let
        new_lcc.cutter1_num = self.cutter1_num
        new_lcc.cutter1_suffix = self.cutter1_suffix
        new_lcc.has_cutter2 = self.has_cutter2
        new_lcc.cutter2_let = self.cutter2_let
        new_lcc.cutter2_num = self.cutter2_num
        new_lcc.cutter2_suffix = self.cutter2_suffix
        new_lcc.has_year = self.has_year
        new_lcc.year = self.year
        new_lcc.year_suffix = self.year_suffix
        new_lcc.has_suffix = self.has_suffix
        new_lcc.suffix_type = self.suffix_type
        new_lcc.suffix_num = self.suffix_num

        # 变异逻辑 (确保 B > A)
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

        # [修复 2] 修正 Cutter 增加逻辑
        elif level == 'cut1':
            # 策略：小数排序中，优先增加末位数字 (如 .C4 -> .C5)
            # 如果末位是 9，则增加位数 (如 .C9 -> .C91)，在小数规则下这确实是变大
            if not self.cutter1_num.endswith('9'):
                # 简单数值增加：.C42 -> .C43 (变大)
                val = int(self.cutter1_num) + 1
                new_lcc.cutter1_num = str(val)
            else:
                # 遇到 9 结尾，通过由 .C9 -> .C91 变大 (小数逻辑)
                new_lcc.cutter1_num += str(random.randint(1, 9))

        elif level == 'cut2':
            if not self.has_cutter2:
                new_lcc.has_cutter2 = True
            else:
                # 同 cut1 逻辑
                if not self.cutter2_num.endswith('9'):
                    val = int(self.cutter2_num) + 1
                    new_lcc.cutter2_num = str(val)
                else:
                    new_lcc.cutter2_num += str(random.randint(1, 9))

        elif level == 'year':
            if not self.has_year:
                new_lcc.has_year = True
                new_lcc.year = random.randint(1980, 2020)
            else:
                new_lcc.year += random.randint(1, 3)

        # [修复 3] 完善 Year Suffix 逻辑
        elif level == 'year_suffix':
            if not self.has_year:
                new_lcc.has_year = True
            
            # 顺序: 无 -> a -> b -> ... -> z -> (年份+1)
            if self.year_suffix == '':
                new_lcc.year_suffix = 'a'
            elif self.year_suffix == 'z':
                new_lcc.year += 1
                new_lcc.year_suffix = '' # 进位后清空后缀
            else:
                # 字符递增 'a' -> 'b'
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

def apply_noise(text):
    """将干净的 LCC 字符串转化为带有 OCR 噪声的版本"""
    chars = list(text)
    noisy_chars = []

    # 随机添加前缀
    if random.random() < 0.05:
        prefix = random.choice(PREFIXES)
        sep = random.choice(['\n', ' '])
        noisy_chars.extend(list(prefix + sep))

    for char in chars:
        if char in OCR_CONFUSION and random.random() < 0.02: # 2% 噪声率
            noisy_chars.append(random.choice(OCR_CONFUSION[char]))
        elif char in [' ', '\n'] and random.random() < 0.05: # 丢失分隔符
            continue
        else:
            noisy_chars.append(char)

        if random.random() < 0.001:
            noisy_chars.append(random.choice(['.', ',', "'"]))

    return "".join(noisy_chars)


def generate_dataset(num_samples=NUM_SAMPLES, output_file=OUTPUT_FILE):
    """生成训练数据集"""
    print(f"Generating {num_samples} samples...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 使用训练代码期望的列名
        writer.writerow(['text_a', 'text_b', 'label'])

        for i in range(num_samples):
            # 随机决定生成哪种类型的数据
            # 0: A < B, 1: A = B, 2: A > B
            data_type = random.choices([0, 1, 2], weights=[0.45, 0.1, 0.45])[0]

            if data_type == 0:
                # A < B: 生成 A 和 B，其中 B = A.clone_and_increment()
                lcc_a = LCCCallNumber()
                lcc_b = lcc_a.clone_and_increment()
                label = 0  # A < B
            elif data_type == 1:
                # A = B: 生成相同的号码
                lcc = LCCCallNumber()
                # 关键修复：使用同一个对象，确保添加噪声后仍然相同
                # 先添加噪声，然后复制结果
                clean_text = lcc.to_string()
                noisy_text = apply_noise(clean_text)
                text_a = noisy_text
                text_b = noisy_text  # 完全相同的文本（包括噪声）
                writer.writerow([text_a, text_b, 1])
                continue  # 跳过后续处理
            else:
                # A > B: 生成 B 和 A，其中 A = B.clone_and_increment()
                lcc_b = LCCCallNumber()
                lcc_a = lcc_b.clone_and_increment()
                label = 2  # A > B

            # 转换为字符串
            label_a = lcc_a.to_string()
            label_b = lcc_b.to_string()

            # 添加 OCR 噪声 - 使用带噪声的版本作为训练数据
            text_a = apply_noise(label_a)
            text_b = apply_noise(label_b)

            # 写入数据
            writer.writerow([text_a, text_b, label])

            # 进度显示
            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i + 1}/{num_samples} samples generated")

    print(f"Dataset saved to {output_file}")
    print(f"Total samples: {num_samples}")


if __name__ == "__main__":
    # 设置随机种子保证可复现性
    random.seed(42)

    # 生成数据集
    generate_dataset()

    # 简单测试验证
    print("\n=== Validation Tests ===")
    test_a = LCCCallNumber()
    test_a.cutter1_num = "9"
    test_a.year = 2000
    test_a.year_suffix = "a"

    print(f"Original: .C{test_a.cutter1_num} {test_a.year}{test_a.year_suffix}")

    test_b = test_a.clone_and_increment()
    print(f"After increment: .C{test_b.cutter1_num} {test_b.year}{test_b.year_suffix}")

    print("\n✓ Data generation completed!")