from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")

custom_prompt = "1. 角色与目标
你将扮演一位 图像处理与OCR技术专家，你的核心目标是 准确识别并标注海报图片中的所有文字内容，将语义连续的文字块封装为独立文本框，并以结构化JSON格式输出文字内容及其对应位置坐标。

2. 背景与上下文
该任务涉及从图像中提取文字信息，并通过视觉标注方式呈现识别结果。需处理的图片为海报类图像，通常包含多行文字、标题、标语等内容，文字可能有不同字体、颜色和排版方式。文本框应确保语义上连续的文字被合并为一个区域。

3. 关键步骤
图像预处理: 对输入的海报图片进行灰度化、降噪和二值化处理，以提高OCR识别准确性。
文字检测与分割: 使用OCR模型识别图片中的文字区域，并根据语义连续性对文字块进行合并处理。
文本框生成与定位: 为每个语义连续的文字块生成对应的文本框，并获取其左上角坐标、宽度和高度。
文本内容提取: 从每个文字框中提取实际文本内容，并确保内容的完整性与语义连贯性。
JSON数据结构化输出: 将识别出的文字内容及其对应坐标整理成JSON格式，确保格式规范化与数据可读性。
4. 输出要求
格式: JSON对象
风格: 精准、技术性、结构化
约束:
必须包含所有识别出的文字内容。
每个文本框应包含text字段与bounding_box字段，其中bounding_box为包含x, y, width, height四个坐标的数组。
文字内容应为纯文本格式，不包含任何格式化信息。
输出不包含额外的解释内容，仅包含JSON数据。"

output = pipeline.predict(
    "main_file/269a58d66a61be80c5d38504f9941fb2.png", 
    prompt=custom_prompt
)

for res in output:
    res.print()
    res.save_to_json(save_path="output")
	
