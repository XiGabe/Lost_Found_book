import json
from typing import List, Dict

class Dataset5Integration:
    """简单的OCR数据聚合器"""

    def load_ocr_file(self, file_path: str) -> List[Dict]:
        """加载OCR结果文件"""
        with open(file_path, 'r') as f:
            content = f.read().strip()

        results = []
        for line in content.split('\n'):
            if line.strip():
                try:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        json_content = parts[1]
                        ocr_data = json.loads(json_content)
                        results.append({
                            'image_file': parts[0],
                            'ocr_data': ocr_data
                        })
                except Exception as e:
                    print(f"解析文件 {file_path} 出错: {e}")

        return results

    def process_dataset5(self) -> Dict:
        """处理dataset5的所有OCR数据，只聚合transcription"""
        files = [
            'e2e_results/dataset5_img_1.txt',
            'e2e_results/dataset5_img_2.txt',
            'e2e_results/dataset5_img_3.txt',
            'e2e_results/dataset5_img_4.txt',
            'e2e_results/dataset5_img_5.txt',
            'e2e_results/dataset5_img_6.txt'
        ]

        all_books = []

        for file_path in files:
            try:
                results = self.load_ocr_file(file_path)
                for result in results:
                    # 只提取transcription和计算基本信息
                    transcriptions = []
                    all_center_x = []
                    all_center_y = []
                    avg_confidence = 0

                    for comp in result['ocr_data']:
                        transcriptions.append(comp['transcription'])
                        # 计算每个组件的中心点
                        points = comp['points']
                        center_x = sum(p[0] for p in points) / 4
                        center_y = sum(p[1] for p in points) / 4
                        all_center_x.append(center_x)
                        all_center_y.append(center_y)
                        avg_confidence += comp['score']

                    # 计算整本书的平均坐标
                    book_center_x = sum(all_center_x) / len(all_center_x)
                    book_center_y = sum(all_center_y) / len(all_center_y)
                    avg_confidence = avg_confidence / len(result['ocr_data'])

                    book_data = {
                        'position': len(all_books) + 1,
                        'image_file': result['image_file'],
                        'transcriptions': transcriptions,
                        'center': [round(book_center_x, 0), round(book_center_y, 0)],
                        'confidence': round(avg_confidence, 3)
                    }
                    all_books.append(book_data)

                    print(f"位置 {book_data['position']}: {book_data['image_file']}")
                    print(f"  内容: {' '.join(transcriptions)}")
            except Exception as e:
                print(f"处理文件 {file_path} 出错: {e}")

        print(f"\n总共处理了 {len(all_books)} 个图像文件")

        # 生成简单报告
        report = {
            'dataset_info': {
                'name': 'Dataset 5',
                'description': '从左到右拍摄的6张书架图片',
                'total_images': len(files),
                'files_processed': len(all_books)
            },
            'books': all_books
        }

        return report

def main():
    """主函数"""
    processor = Dataset5Integration()
    report = processor.process_dataset5()

    # 保存报告
    with open('Data_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 输出摘要
    print(f"\n{'='*60}")
    print("DATASET 5 OCR数据聚合报告")
    print(f"{'='*60}")
    print(f"总图像数: {report['dataset_info']['total_images']}")
    print(f"处理文件数: {report['dataset_info']['files_processed']}")
    print(f"\n详细报告已保存到: dataset5_analysis_report.json")

if __name__ == "__main__":
    main()