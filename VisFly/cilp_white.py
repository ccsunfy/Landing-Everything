import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import os

def remove_white_background_pil(input_path, output_path, threshold=240):
    """
    去除图片白色背景，保持高质量
    """
    try:
        # 打开图片并转换为RGBA模式
        img = Image.open(input_path).convert("RGBA")
        
        # 获取原始尺寸
        original_size = img.size
        
        # 将图片转换为numpy数组进行处理
        data = np.array(img)
        
        # 创建白色掩码
        white_mask = (data[:, :, 0] > threshold) & \
                     (data[:, :, 1] > threshold) & \
                     (data[:, :, 2] > threshold)
        
        # 将白色区域的Alpha通道设置为0（完全透明）
        data[:, :, 3] = np.where(white_mask, 0, 255)
        
        # 转换回PIL图像
        result_img = Image.fromarray(data, "RGBA")
        
        # 保持原始尺寸保存
        result_img.save(output_path, "PNG", optimize=True)
        
        print(f"已处理: {output_path} (尺寸: {original_size})")
        
    except Exception as e:
        print(f"处理图片时出错: {e}")

def pdf_to_high_quality_images(pdf_path, output_folder, dpi=300):
    """
    将PDF文件转换为高质量图片
    
    参数:
        pdf_path: PDF文件路径
        output_folder: 输出文件夹
        dpi: 图片分辨率，越高越清晰
    """
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = []
    
    # 遍历每一页
    for page_num in range(len(pdf_document)):
        # 获取页面
        page = pdf_document[page_num]
        
        # 设置高分辨率转换矩阵
        zoom = dpi / 72  # 72是默认的DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # 将页面转换为高质量图片
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # 保存为PNG图片
        output_path = os.path.join(output_folder, f"page_{page_num+1}.png")
        pix.save(output_path)
        image_paths.append(output_path)
        
        print(f"已转换第 {page_num+1} 页 (分辨率: {pix.width} x {pix.height})")
    
    pdf_document.close()
    return image_paths

def process_pdf_high_quality(pdf_path, output_folder, dpi=300, threshold=240):
    """
    高质量处理PDF：转换为高清图片并去除白色背景
    """
    try:
        # 1. 将PDF转换为高质量图片
        print("正在转换PDF为高质量图片...")
        temp_folder = os.path.join(output_folder, "temp_high_quality")
        image_paths = pdf_to_high_quality_images(pdf_path, temp_folder, dpi)
        
        # 2. 处理每张图片的白色背景
        print("正在去除白色背景...")
        processed_paths = []
        
        for i, img_path in enumerate(image_paths):
            # 处理白色背景
            processed_path = os.path.join(output_folder, f"高清_无背景_第{i+1}页.png")
            remove_white_background_pil(img_path, processed_path, threshold)
            processed_paths.append(processed_path)
            
            # 删除临时图片
            os.remove(img_path)
        
        # 删除临时文件夹
        os.rmdir(temp_folder)
        
        print(f"高质量处理完成！共处理 {len(processed_paths)} 页")
        return processed_paths
        
    except Exception as e:
        print(f"处理PDF时出错: {e}")
        return []

# 使用示例
if __name__ == "__main__":
    pdf_file = "d435i.jpg"
    output_dir = "vicon_processed"
    
    # 使用高DPI (300-600) 获得高清图片
    process_pdf_high_quality(pdf_file, output_dir, dpi=300, threshold=240)