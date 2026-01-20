import os
import time
import json
import jwt
import requests


BASE_URL = "https://mhapi.sensetime.com"
MODELS_PATH = "/v1/imgenstd/models"


def encode_jwt_token(ak: str, sk: str, expire_seconds: int = 1800) -> str:
    """
    使用 AK/SK 生成 JWT token（HS256）
    payload:
      - iss: AK
      - exp: 当前时间 + expire_seconds
      - nbf: 当前时间 - 5（防止时间误差）
    """
    now = int(time.time())

    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": now + expire_seconds,
        "nbf": now - 5,
    }

    token = jwt.encode(payload, sk, algorithm="HS256", headers=headers)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def fetch_models_page(token: str, size: int = 100, offset: int = 0, mtp: str = "ALL") -> dict:
    """
    获取一页模型列表
    """
    url = BASE_URL + MODELS_PATH
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    params = {
        "size": size,
        "offset": offset,
        "mtp": mtp,  # "LORA" / "Checkpoint" / "ALL"
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)

    # 如果鉴权失败，直接把错误打印出来
    if resp.status_code != 200:
        raise RuntimeError(
            f"请求失败：HTTP {resp.status_code}\n"
            f"URL: {resp.url}\n"
            f"响应内容: {resp.text}"
        )

    return resp.json()


def fetch_all_models(ak: str, sk: str, size: int = 100, mtp: str = "ALL") -> list:
    """
    自动分页拉取所有模型（直到 have_next == false）
    """
    token = encode_jwt_token(ak, sk)
    all_models = []
    offset = 0

    while True:
        data = fetch_models_page(token=token, size=size, offset=offset, mtp=mtp)

        # 文档返回格式：{"have_next": bool, "data": [ ... ]}
        page_models = data.get("data", [])
        have_next = data.get("have_next", False)

        all_models.extend(page_models)

        print(f"✅ offset={offset} 拉取到 {len(page_models)} 个模型，累计 {len(all_models)} 个")

        if not have_next:
            break

        offset += size

    return all_models


def main():
    # 用环境变量更安全：SENSENOVA_AK / SENSENOVA_SK
    ak = os.getenv("SENSENOVA_AK", "").strip()
    sk = os.getenv("SENSENOVA_SK", "").strip()

    if not ak or not sk:
        print("❌ 未检测到 AK/SK")
        return

    # 参数可调整
    size = 100
    mtp = "ALL"  # "LORA" / "Checkpoint" / "ALL"

    models = fetch_all_models(ak=ak, sk=sk, size=size, mtp=mtp)

    # 输出目录
    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "models.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已保存模型列表：{out_path}")
    print(f"✅ 总模型数量：{len(models)}")

    # 顺便打印前 5 个模型的关键信息
    print("\n📌 前 5 个模型预览：")
    for m in models[:5]:
        print({
            "id": m.get("id"),
            "name": m.get("name"),
            "model_type": m.get("model_type"),
            "base_model_type": m.get("base_model_type"),
        })


if __name__ == "__main__":
    main()
