from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="w5ywe8uErJb2VZkDqh9s"
)

path = "../assets/v6.png"

result = CLIENT.infer(path, model_id="bac_hien_construction_safety_2024/8")

print(result)