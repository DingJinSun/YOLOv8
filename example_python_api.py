# -*- coding: utf-8 -*-
"""YOLOv8 (ultralytics 8.0.29) Python API 使用示例。

运行方式（任选其一）：
    . "E:\\yolo v8\\env.ps1"  然后:  py example_python_api.py
    或直接:  & "E:\\yolo v8\\.venv\\Scripts\\python.exe" "E:\\yolo v8\\YOLOv8\\example_python_api.py"
"""
import torch
from ultralytics import YOLO

print(f"torch {torch.__version__}   CUDA可用: {torch.cuda.is_available()}")

# ------------------------------------------------------------------
# 1. 加载模型（.pt = 预训练权重；.yaml = 从零构建空模型）
#    权重文件放在本项目目录下即可，不会联网（GitHub 在本机不可达）
# ------------------------------------------------------------------
model = YOLO("yolov8n.pt")
names = model.model.names  # 类别名 {0: 'person', 5: 'bus', ...}

# ------------------------------------------------------------------
# 2. 推理：单张图片
# ------------------------------------------------------------------
results = model.predict(source="ultralytics/assets/bus.jpg", save=True, conf=0.25, device=0)
r = results[0]
print(f"\n检出 {len(r.boxes)} 个目标:")
for b in r.boxes:
    box = [round(v) for v in b.xyxy[0].tolist()]  # 先转成 python float 再取整
    print(f"  {names[int(b.cls)]:<12} conf={float(b.conf):.2f}  box={box}")

# ------------------------------------------------------------------
# 3. 推理：整个文件夹 / 视频 / 摄像头 / 网络流
#    source 支持: 图片路径、目录、.mp4 视频、0(摄像头)、http 流地址、PIL/numpy 图像
# ------------------------------------------------------------------
# results = model.predict(source="D:/我的图片/", save=True)      # 目录批量
# results = model.predict(source="D:/视频.mp4", save=True)       # 视频(输出mp4)
# results = model.predict(source=0, show=True, stream=True)      # 摄像头实时窗口

# ------------------------------------------------------------------
# 4. 流式推理（处理大量图片/视频时省内存，逐帧 yield）
# ------------------------------------------------------------------
# for r in model.predict(source="ultralytics/assets", stream=True):
#     boxes = r.boxes.xyxy.cpu().numpy()   # numpy 数组
#     confs = r.boxes.conf.cpu().numpy()

# ------------------------------------------------------------------
# 5. 常用参数（predict）
#    save=True        保存画好框的结果图/视频到 runs\detect\predict*
#    save_txt=True    额外保存 YOLO 格式的 .txt 标注
#    save_crop=True   保存每个目标的裁剪小图
#    save_conf=True   txt 里带置信度
#    conf=0.25        置信度阈值        iou=0.7    NMS 阈值
#    imgsz=640        推理分辨率        device=0   指定 GPU(或 cpu)
#    classes=[0,2]    只保留这些类别    half=True  FP16 加速
#    show=True        弹窗实时显示
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 6. 训练 / 验证 / 导出（示例，按需取消注释）
#    注意：数据集需自行准备，且不要依赖自动下载（GitHub 不可达）
# ------------------------------------------------------------------
# model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8, device=0)
# model.val(data="coco128.yaml")          # np.trapz 已改为 np.trapezoid，可正常跑 val
# model.export(format="onnx")             # 导出 ONNX（需另装 onnx 包）

print("\n结果已保存到 runs\\detect\\predict* 目录")
