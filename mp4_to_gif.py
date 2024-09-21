from PIL import Image

def extract_last_frame(gif_path, output_path):
    with Image.open(gif_path) as img:
        img.seek(img.n_frames - 1)  # 移动到最后一帧
        last_frame = img.copy()  # 复制最后一帧
        last_frame.save(output_path, 'PNG')  # 保存为PNG

# 使用示例
extract_last_frame('./output_1.gif', 'zikai_1.png')
extract_last_frame('./output_2.gif', 'zikai_2.png')
extract_last_frame('./output_3.gif', 'zikai_3.png')
extract_last_frame('./output_4.gif', 'zikai_4.png')
