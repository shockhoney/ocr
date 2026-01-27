import cv2
import numpy as np
from typing import Dict, List
import math
import os
import sys
import dashscope

def get_ocr_result(local_path):
    # 将xxxx/test.png替换为您本地图像的绝对路径
    image_path = f"file://{local_path}"
    messages = [{
        "role": "user",
        "content": [{
            "image": image_path,
            # 输入图像的最小像素阈值，小于该值图像会按原比例放大，直到总像素大于min_pixels
            "min_pixels": 28 * 28 * 4,
            # 输入图像的最大像素阈值，超过该值图像会按原比例缩小，直到总像素低于max_pixels
            "max_pixels": 28 * 28 * 8192,
            # 开启图像自动转正功能
            "enable_rotate": True},
            # 当ocr_options中的task字段设置为高精识别时，模型会以下面text字段中的内容作为Prompt，不支持用户自定义
            {"text": '''定位所有的文字行，并且以json格式返回旋转矩形([x1, y1, x2, y2])的坐标结果。 返回格式必须是纯 JSON 列表，格式如下：
           [
             {"text": "文本内容", "box": [xmin, ymin, xmax, ymax]},
             ...
           ]其中x1,y1是文字行的左上角坐标，x2,y2是文字行的右下角坐标'''}]
    }]

    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-7f9cfd5f5f154c13b03604128a0b5491",
        model='qwen-vl-ocr-2025-08-28',
        messages=messages,
        # 设置内置任务为高精识别
        ocr_options={"task": "advanced_recognition"}
    )

    ocr_result=response["output"]["choices"][0]["message"].content[0]["ocr_result"]
    return ocr_result



def _rrect_to_box(rrect) -> List:
    x, y, width, height, angle = rrect
    angle = angle % 180
    angle_rad = math.radians(angle)

    # 四个未旋转的顶点（相对于中心）
    corners = [
        (-width / 2, -height / 2),  # 左上
        (width / 2, -height / 2),  # 右上
        (width / 2, height / 2),  # 右下
        (-width / 2, height / 2)  # 左下
    ]

    # 旋转并平移到最终位置
    rotated_corners = []
    for (px, py) in corners:
        # 二维旋转公式
        x_new = x + px * math.cos(angle_rad) - py * math.sin(angle_rad)
        y_new = y + px * math.sin(angle_rad) + py * math.cos(angle_rad)
        rotated_corners.append(x_new)
        rotated_corners.append(y_new)

    # 转换为 int32 类型
    result = np.array(rotated_corners, dtype=np.int32)

    return result.tolist()


def print_location(image_file='', ocr_result={}):
    print(f'input image_file=[{image_file}]')
    print(f'input ocr_result=[{ocr_result}]')
    try:
        if 'words_info' not in ocr_result:
            print(f'no words_info result')
            sys.exit(0)
        words_info = ocr_result["words_info"]

        img = cv2.imread(image_file)

        for line in words_info:
            x1, y1, x2, y2, x3, y3, x4, y4 = line['location']
            cv2.line(img, (int(float(x1)), int(float(y1))), (int(float(x2)), int(float(y2))), (0, 0, 255), 2)
            cv2.line(img, (int(float(x2)), int(float(y2))), (int(float(x3)), int(float(y3))), (0, 0, 255), 2)
            cv2.line(img, (int(float(x3)), int(float(y3))), (int(float(x4)), int(float(y4))), (0, 0, 255), 2)
            cv2.line(img, (int(float(x4)), int(float(y4))), (int(float(x1)), int(float(y1))), (0, 0, 255), 2)
        file_name = os.path.basename(image_file)
        base_name = os.path.splitext(file_name)[0]
        ext_name = os.path.splitext(image_file)[1]
        dir_name = os.path.dirname(image_file)
        print(f'file_name={file_name}   dir_name={dir_name}')
        save_vis_name = dir_name + "/" + base_name + "_location" + ext_name
        print(f'save image to {save_vis_name}')

        # 保存已绘制检测框的图像
        cv2.imwrite(save_vis_name, img)
        cv2.imshow('OCR Result', img)
        cv2.waitKey(0)  # 按下任意键，即可关闭绘图窗口
        cv2.destroyAllWindows()
    except Exception as e:
        print(f'catch exception {image_file}:{ocr_result}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 本地图像的路径
    image_file = 'main_file/2a005f011d4e8bf1a3d55dfd9837eba2.png'
    ocr_result= get_ocr_result(image_file)
    print_location(image_file, ocr_result)

