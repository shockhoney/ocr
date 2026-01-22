from paddleocr import PaddleOCRVL
# 指向启动的vllm服务
pipeline = PaddleOCRVL(vl_rec_backend = "vllm-server",vl_rec_server_url = "http://127.0.0.1:8118/v1")

output = pipeline.predict("main_file/269a58d66a61be80c5d38504f9941fb2.png")

for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
