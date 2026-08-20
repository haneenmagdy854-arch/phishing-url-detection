import requests
import sys

URL = "http://127.0.0.1:5000/api/analyze"
payload = {"url": "https://paypal-login-verification.com"}

try:
    resp = requests.post(URL, json=payload, timeout=10)
    resp.raise_for_status()
except requests.exceptions.RequestException as e:
    print("❌ خطأ في الاتصال أو الاستعلام:", e)
    sys.exit(1)

try:
    data = resp.json()
except ValueError:
    print("❌ الرد ليس JSON:", resp.text)
    sys.exit(1)

print("=== API Response ===")
for k, v in data.items():
    print(f"{k}: {v}")