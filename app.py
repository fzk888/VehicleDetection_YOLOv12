import gradio as gr
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO(r"VehicleDetection_YOLOv12\exp15\weights\best.pt")

def predict_image(image):
    """
    对输入的图像进行目标检测
    :param image: 输入的图像
    :return: 检测后的图像
    """
    results = model.predict(image, save=False, show=False)
    annotated_image = results[0].plot()
    return annotated_image

# 创建Gradio界面
#        # <div style="color: #666; margin-bottom: 20px;">
        #     <span>By AI应用开发助手</span>
        # </div>
with gr.Blocks(title="🚗 车辆检测分类系统", css=".gradio-container {background: #f9f9f9}") as demo:
    # 标题部分
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>🚗 车辆检测分类系统</h1>
        <h3>基于YOLOv12的车辆检测与分类</h3>

    </div>
    """)

    # 主要功能区域
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown("### 📤 图片输入区")
            image_input = gr.Image(label="请上传图片", interactive=True)
            gr.Markdown("""
            <div style="color: #888; font-size: 0.9em;">
                <span>📌 支持格式：JPG/PNG</span><br>
                <span>📌 最大文件大小：10MB</span>
            </div>
            """)
            detect_button = gr.Button("🔍 检测车辆", elem_classes="custom-button")

        with gr.Column(scale=1):
            gr.Markdown("### 📤 检测结果")
            image_output = gr.Image(label="检测后的图片", interactive=False)
            gr.Markdown("""
            <div style="color: #888; font-size: 0.9em; margin-top: 10px;">
                <span>✅ 检测完成后会自动显示结果</span><br>
                <span>⏳ 检测时间取决于图片大小和复杂度</span>
            </div>
            """)

    # 设置按钮点击事件
    detect_button.click(
        fn=predict_image,
        inputs=image_input,
        outputs=image_output
    )

    # 页脚说明
    gr.Markdown("---")
    gr.Markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>🛠️ 技术支持：OpenCV | Gradio | YOLO</p>
        <p>⚠️ 注意事项：复杂图片可能需要较长时间处理，请耐心等待</p>
    </div>
    """)

# 添加自定义CSS样式
css = """
.custom-button {
    background: linear-gradient(45deg, #4CAF50, #45a049);
    color: white;
    border: none;
    padding: 15px 30px;
    border-radius: 8px;
    font-weight: bold;
    transition: all 0.3s;
}
.custom-button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
"""
demo.css = css

# 启动Gradio应用
demo.launch(share=False)