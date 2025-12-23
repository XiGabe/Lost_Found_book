"""
TensorRT Engine 推理性能测试
测试推理速度并保存最后 N 张检测结果
"""
import os
import time
import glob
import cv2
import numpy as np
from ultralytics import YOLO


# ==================== 配置 ====================
MODEL_PATH = 'best.engine'
SOURCE_DIR = 'data/images'
OUTPUT_DIR = 'runs/benchmark_results'
IMG_SIZE = 640
CONF = 0.25
IOU = 0.45
DEVICE = 0
WARMUP = 5
SAVE_LAST_N = 5
# ===============================================


def benchmark():
    # 检查文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        return

    # 获取图片列表
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))

    if not images:
        print(f"❌ 未找到图片: {SOURCE_DIR}")
        return

    print(f"📂 图片数量: {len(images)}")
    print(f"🔥 预热 {WARMUP} 轮...")

    # 加载模型
    model = YOLO(MODEL_PATH, task='segment')

    # 预热
    for i in range(WARMUP):
        model.predict(images[i % len(images)], imgsz=IMG_SIZE, device=DEVICE, verbose=False)

    print("⚡ 开始推理...")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 性能数据
    times = {'preprocess': [], 'inference': [], 'postprocess': []}
    last_results = []

    # 推理
    t_start = time.time()
    for i, r in enumerate(model.predict(
        source=SOURCE_DIR,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        save=False,
        verbose=False,
        stream=True
    )):
        times['preprocess'].append(r.speed['preprocess'])
        times['inference'].append(r.speed['inference'])
        times['postprocess'].append(r.speed['postprocess'])

        # 保存最后 N 张
        if i >= len(images) - SAVE_LAST_N:
            last_results.append(r)

        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(images)}")

    t_total = time.time() - t_start

    # 保存结果图片
    print(f"\n💾 保存最后 {len(last_results)} 张到 {OUTPUT_DIR}/")
    for r in last_results:
        img = r.cpu().plot()
        cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(r.path)), img)

    # 统计
    n = len(images)
    avg_prep = np.mean(times['preprocess'])
    avg_inf = np.mean(times['inference'])
    avg_post = np.mean(times['postprocess'])

    print("\n" + "=" * 45)
    print("🎉 性能测试完成")
    print("=" * 45)
    print(f"  图片数量:    {n}")
    print(f"  总耗时:      {t_total:.2f} s")
    print(f"  保存结果:    {OUTPUT_DIR}/ ({len(last_results)} 张)")
    print(f"\n  平均耗时:")
    print(f"    预处理:    {avg_prep:.2f} ms")
    print(f"    推理:      {avg_inf:.2f} ms")
    print(f"    后处理:    {avg_post:.2f} ms")
    print(f"    总计:      {avg_prep + avg_inf + avg_post:.2f} ms")
    print(f"\n  吞吐量:")
    print(f"    端到端:    {n / t_total:.2f} FPS")
    print(f"    纯推理:    {1000 / avg_inf:.2f} FPS")
    print("=" * 45)


if __name__ == '__main__':
    benchmark()
