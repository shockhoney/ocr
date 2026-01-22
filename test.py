from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")

custom_prompt ='''1. 角色与目标
你将扮演一位 图像处理与OCR技术专家，你的核心目标是 准确识别并标注海报图片中的所有文字内容，将语义连续的文字块封装为独立文本框，并以结构化JSON格式输出文字内容及其对应位置坐标。

2. 背景与上下文
该任务涉及从图像中提取文字信息，并通过视觉标注方式呈现识别结果。需处理的图片为海报类图像，
通常包含多行文字、标题、标语等内容，文字可能有不同字体、颜色和排版方式。文本框应确保语义上连续的文字被合并为一个区域。
3. 输出要求
格式: JSON对象
风格: 精准、技术性、结构化
约束:
必须包含所有识别出的文字内容。
每个文本框应包含text字段与bounding_box字段，其中bounding_box为包含x, y, width, height四个坐标的数组。
文字内容应为纯文本格式，不包含任何格式化信息。
输出不包含额外的解释内容，仅包含JSON数据。'''

output = pipeline.predict(
    "main_file/	2ce0c5c86d4b290fac62168e3bb48392.png", 
    prompt=custom_prompt
)

for res in output:
    res.print()
    res.save_to_json(save_path="output")
	
