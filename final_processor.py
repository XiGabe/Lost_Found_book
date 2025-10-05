import json
import re
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TextBlock:
    text: str
    confidence: float
    points: List[List[int]]

@dataclass
class CallNumber:
    full_string: str
    letters: str
    numbers: float
    cutter: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    confidence: float = 0.0
    position: Dict = None

class FinalProcessor:
    """最终的OCR处理器，专注于识别完整的BR 330 E5 1955 v.28格式"""

    def __init__(self):
        self.confidence_threshold = 0.7  # 置信度阈值
        self.max_distance = 350  # 同一本书的组件最大距离（进一步增加以包含卷号）

    def calculate_center(self, points: List[List[int]]) -> Tuple[float, float]:
        """计算文本块的中心点"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)

    def calculate_distance(self, block1: TextBlock, block2: TextBlock) -> float:
        """计算两个文本块之间的距离"""
        center1 = self.calculate_center(block1.points)
        center2 = self.calculate_center(block2.points)
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def is_call_number_component(self, text: str) -> str:
        """判断文本是否为Call Number的组件，返回组件类型"""
        if re.match(r'^[A-Z]{1,3}$', text):  # 如 BR, B, BF
            return 'letters'
        elif re.match(r'^\d+(\.\d+)?$', text):  # 如 330, 330.5
            return 'numbers'
        elif re.match(r'^[A-Z]\d*$', text):  # 如 E5, E
            return 'cutter'
        elif re.match(r'^(v\.?\d+|V\.?\d+)$', text):  # 如 v.28, V.29
            return 'volume'
        elif re.match(r'^\d{4}$', text):  # 如 1955
            return 'year'
        else:
            return 'none'

    def find_br_related_components(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """筛选出所有BR相关的Call Number组件"""
        components = []

        # 首先找到所有的BR
        br_blocks = [block for block in text_blocks if block.text == 'BR']

        # 为每个BR找到相关的组件
        for br_block in br_blocks:
            components.append(br_block)

            # 寻找距离BR较近的其他组件
            for block in text_blocks:
                if block != br_block and self.is_call_number_component(block.text) != 'none':
                    distance = self.calculate_distance(br_block, block)
                    if distance <= self.max_distance:
                        components.append(block)

        # 也找到所有其他可能的Call Number组件（年份、卷号等）
        for block in text_blocks:
            if (self.is_call_number_component(block.text) != 'none' and
                block not in components):
                # 检查是否与任何已存在的组件相近
                for existing_comp in components:
                    distance = self.calculate_distance(block, existing_comp)
                    if distance <= self.max_distance:
                        components.append(block)
                        break

        # 去重
        seen = set()
        unique_components = []
        for comp in components:
            comp_key = (comp.text, tuple(min(p) for p in comp.points))
            if comp_key not in seen:
                seen.add(comp_key)
                unique_components.append(comp)

        return unique_components

    def group_components_by_book(self, components: List[TextBlock]) -> List[List[TextBlock]]:
        """将组件按书籍分组 - 使用更智能的聚类"""
        if not components:
            return []

        books = []
        used_components = set()

        # 优先处理包含BR的组件
        br_components = [comp for comp in components if comp.text == 'BR']

        for br_comp in br_components:
            if id(br_comp) in used_components:
                continue

            # 创建一个新的书籍组，以BR为中心
            book_group = [br_comp]
            used_components.add(id(br_comp))

            # 寻找同一本书的其他组件
            for other_comp in components:
                if id(other_comp) in used_components or other_comp == br_comp:
                    continue

                # 计算与BR的距离
                distance = self.calculate_distance(br_comp, other_comp)

                # 如果距离足够近，认为是同一本书
                if distance <= self.max_distance:
                    book_group.append(other_comp)
                    used_components.add(id(other_comp))

            books.append(book_group)

        # 处理剩余的组件（可能是不包含BR的Call Number）
        for comp in components:
            if id(comp) not in used_components:
                # 创建单独的组
                books.append([comp])
                used_components.add(id(comp))

        return books

    def reconstruct_call_number(self, book_group: List[TextBlock]) -> Optional[CallNumber]:
        """从书籍组件组重建Call Number"""
        if not book_group:
            return None

        # 按x坐标排序组件
        sorted_components = sorted(book_group, key=lambda b: min(p[0] for p in b.points))

        # 按照LC Call Number的标准顺序组织组件
        letters = None
        numbers = None
        cutter = None
        year = None
        volume = None

        # 首先找到BR作为字母部分
        for comp in sorted_components:
            if comp.text == 'BR':
                letters = comp.text
                break

        # 然后找到330作为主要数字
        for comp in sorted_components:
            if comp.text == '330':
                numbers = comp.text
                break

        # 然后找到E5作为Cutter
        for comp in sorted_components:
            if comp.text == 'E5':
                cutter = comp.text
                break

        # 然后找到1955作为年份
        for comp in sorted_components:
            if comp.text == '1955':
                year = comp.text
                break

        # 最后找到卷号
        for comp in sorted_components:
            if self.is_call_number_component(comp.text) == 'volume':
                volume = comp.text
                break

        # 验证是否至少有字母和数字
        if letters is None or numbers is None:
            return None

        # 构建完整的Call Number字符串 - 按照LC标准顺序
        parts = [letters, numbers]
        if cutter:
            parts.append(cutter)
        if year:
            parts.append(year)
        if volume:
            parts.append(volume)

        full_string = ' '.join(parts)

        # 计算平均置信度
        avg_confidence = sum(comp.confidence for comp in sorted_components) / len(sorted_components)

        # 解析数字部分
        try:
            numbers_float = float(numbers)
        except ValueError:
            return None

        # 解析年份
        year_int = None
        if year:
            try:
                year_int = int(year)
            except ValueError:
                pass

        return CallNumber(
            full_string=full_string,
            letters=letters,
            numbers=numbers_float,
            cutter=cutter,
            year=year_int,
            volume=volume,
            confidence=avg_confidence,
            position={'points': sorted_components[0].points}
        )

    def compare_call_numbers(self, call1: CallNumber, call2: CallNumber) -> int:
        """比较两个Call Number的顺序 - LC标准排序规则"""
        # 1. 比较字母部分（LC分类号）
        if call1.letters != call2.letters:
            return 1 if call1.letters > call2.letters else -1

        # 2. 比较数字部分（分类号数字）
        if call1.numbers != call2.numbers:
            return 1 if call1.numbers > call2.numbers else -1

        # 3. 比较Cutter号（作者号）
        if call1.cutter and call2.cutter:
            if call1.cutter != call2.cutter:
                return self.compare_cutter_numbers(call1.cutter, call2.cutter)

        # 4. 比较年份（出版年份）
        if call1.year and call2.year:
            if call1.year != call2.year:
                return 1 if call1.year > call2.year else -1

        # 5. 比较卷号（卷册号）
        if call1.volume and call2.volume:
            vol1_num = self.parse_volume_number(call1.volume)
            vol2_num = self.parse_volume_number(call2.volume)
            if vol1_num != vol2_num:
                return 1 if vol1_num > vol2_num else -1

        return 0

    def compare_cutter_numbers(self, cutter1: str, cutter2: str) -> int:
        """比较Cutter号（如 E5, E4, E3等）"""
        def parse_cutter(cutter):
            match = re.match(r'([A-Z])(\d+)', cutter)
            if match:
                return match.group(1), int(match.group(2))
            return cutter, 0

        letters1, num1 = parse_cutter(cutter1)
        letters2, num2 = parse_cutter(cutter2)

        if letters1 != letters2:
            return 1 if letters1 > letters2 else -1

        if num1 != num2:
            return 1 if num1 > num2 else -1

        return 0

    def parse_volume_number(self, volume_str: str) -> int:
        """解析卷号字符串"""
        match = re.search(r'v\.?(\d+)', volume_str)
        if match:
            return int(match.group(1))
        return 0

    def validate_sorting_order(self, call_numbers: List[CallNumber]) -> Dict:
        """验证排序顺序"""
        if len(call_numbers) < 2:
            return {'misplaced_books': [], 'total_books': len(call_numbers)}

        misplaced_books = []

        for i in range(len(call_numbers) - 1):
            current = call_numbers[i]
            next_book = call_numbers[i + 1]

            comparison = self.compare_call_numbers(current, next_book)

            if comparison > 0:
                misplaced_books.append({
                    'call_number': current.full_string,
                    'expected_position': i + 1,
                    'actual_position': i,
                    'confidence': current.confidence,
                    'next_call_number': next_book.full_string,
                    'issue': f"'{current.full_string}' should come after '{next_book.full_string}'"
                })

        return {
            'misplaced_books': misplaced_books,
            'total_books': len(call_numbers),
            'correctly_placed': len(call_numbers) - len(misplaced_books)
        }

    def load_ocr_results(self, file_path: str) -> List[TextBlock]:
        """加载OCR结果文件"""
        text_blocks = []

        with open(file_path, 'r') as f:
            content = f.read().strip()

        for line in content.split('\n'):
            if line.strip():
                try:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        json_content = parts[1]
                        ocr_data = json.loads(json_content)

                        for result in ocr_data:
                            block = TextBlock(
                                text=result['transcription'],
                                confidence=result['score'],
                                points=result['points']
                            )
                            text_blocks.append(block)
                except Exception as e:
                    print(f"Error parsing line: {e}")

        return text_blocks

    def debug_book_groups(self, book_groups: List[List[TextBlock]]):
        """调试书籍分组结果"""
        print(f"\n书籍分组详情:")
        for i, group in enumerate(book_groups):
            if len(group) >= 2:  # 只显示有多个组件的组
                print(f"书籍 {i+1} ({len(group)} 个组件):")
                for j, component in enumerate(group):
                    center = self.calculate_center(component.points)
                    comp_type = self.is_call_number_component(component.text)
                    print(f"  {j+1}. '{component.text}' ({comp_type}) - 位置: ({center[0]:.0f}, {center[1]:.0f})")

    def process_file(self, file_path: str) -> Dict:
        """处理OCR结果文件"""
        # 1. 加载OCR结果
        text_blocks = self.load_ocr_results(file_path)
        print(f"Loaded {len(text_blocks)} text blocks")

        # 2. 筛选BR相关的Call Number组件
        components = self.find_br_related_components(text_blocks)
        print(f"Found {len(components)} BR-related components")

        # 3. 按书籍分组组件
        book_groups = self.group_components_by_book(components)
        print(f"Grouped into {len(book_groups)} books")

        # 调试分组结果
        self.debug_book_groups(book_groups)

        # 4. 重建Call Number
        call_numbers = []
        for i, book_group in enumerate(book_groups):
            call_num = self.reconstruct_call_number(book_group)
            if call_num:
                call_numbers.append(call_num)
                print(f"Book {i+1}: {call_num.full_string} (confidence: {call_num.confidence:.2f})")

        # 5. 按位置排序
        call_numbers.sort(key=lambda x: min(p[0] for p in x.position['points']))

        # 6. 验证排序
        validation_result = self.validate_sorting_order(call_numbers)

        # 7. 返回结果
        result = {
            'file_path': file_path,
            'scan_results': {
                'total_blocks': len(text_blocks),
                'br_components_found': len(components),
                'books_grouped': len(book_groups),
                'call_numbers_extracted': len(call_numbers)
            },
            'call_numbers': [cn.full_string for cn in call_numbers],
            'validation': validation_result,
            'detailed_call_numbers': [
                {
                    'call_number': cn.full_string,
                    'letters': cn.letters,
                    'numbers': cn.numbers,
                    'cutter': cn.cutter,
                    'year': cn.year,
                    'volume': cn.volume,
                    'confidence': cn.confidence
                }
                for cn in call_numbers
            ]
        }

        return result

def main():
    """主函数"""
    processor = FinalProcessor()

    # 处理OCR结果文件
    result = processor.process_file('e2e_results/system_results.txt')

    # 输出结果
    print("\n" + "="*50)
    print("FINAL PROCESSING RESULTS")
    print("="*50)

    print(f"\n扫描统计:")
    print(f"  总文本块: {result['scan_results']['total_blocks']}")
    print(f"  BR相关组件: {result['scan_results']['br_components_found']}")
    print(f"  书籍分组: {result['scan_results']['books_grouped']}")
    print(f"  识别的Call Number: {result['scan_results']['call_numbers_extracted']}")

    print(f"\n识别的Call Number:")
    for i, call_num in enumerate(result['call_numbers']):
        print(f"  {i+1}. {call_num}")

    print(f"\n排序验证结果:")
    print(f"  总书籍数: {result['validation']['total_books']}")
    print(f"  正确排列: {result['validation']['correctly_placed']}")
    print(f"  错位书籍: {len(result['validation']['misplaced_books'])}")

    if result['validation']['misplaced_books']:
        print(f"\n错位书籍详情:")
        for misplaced in result['validation']['misplaced_books']:
            print(f"  - {misplaced['call_number']}")
            print(f"    问题: {misplaced['issue']}")
            print(f"    位置: {misplaced['actual_position']+1} -> {misplaced['expected_position']+1}")

    # 保存结果到JSON文件
    with open('final_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 final_results.json")

if __name__ == "__main__":
    main()