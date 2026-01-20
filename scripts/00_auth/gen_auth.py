import time
import jwt  

# ✅ 这里填你控制台生成的 AK 和 SK
AK = "019BD41896A575EBAB2F42FC97869C01"
SK = "019BD41896A575DDA51912F63CFD18CA"

def encode_jwt_token(ak: str, sk: str, expire_seconds: int = 1800) -> str:
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }

    now = int(time.time())

    payload = {
        "iss": ak,
        "exp": now + expire_seconds,  # 过期时间：当前时间+1800秒（30分钟）
        "nbf": now - 5,               # 生效时间：当前时间-5秒（防止时间误差）
    }

    token = jwt.encode(payload, sk, algorithm="HS256", headers=headers)

    # PyJWT 有时返回 bytes，保险处理一下
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return token

token = encode_jwt_token(AK, SK)

print("\n✅ 生成的 YOUR_TOKEN：\n")
print(token)

print("\n✅ 你请求头应该这样写：\n")
print("Authorization: Bearer " + token)
