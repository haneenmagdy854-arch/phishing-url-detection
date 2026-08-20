# debug_model.py
import joblib
from pprint import pprint
from urllib.parse import urlparse
import re
import ipaddress
import os
import sys

# ---------- ثابتات ودالة استخراج الميزات (مطابقة للتدريب وapp.py) ----------
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'account', 'update', 'secure',
    'banking', 'ebay', 'paypal', 'amazon', 'facebook', 'google', 'apple', 'microsoft'
]
URL_SHORTENERS = [
    'bit.ly', 't.co', 'tinyurl.com', 'goo.gl', 'ow.ly', 'buff.ly', 't.ly'
]

def extract_features(url):
    features = {}
    if not isinstance(url, str) or url.strip() == '':
        return {}
    url = url.strip()
    # ensure scheme for parsing
    if not re.match(r'^(http|https)://', url, flags=re.I):
        url = 'http://' + url
    try:
        parsed = urlparse(url)
        domain = parsed.hostname
        if domain is None:
            return {}
        # uses_ip
        try:
            ipaddress.ip_address(domain)
            features['uses_ip'] = 1
        except ValueError:
            features['uses_ip'] = 0
        features['url_length'] = len(url)
        features['has_at_symbol'] = 1 if '@' in url else 0
        parts = domain.split('.')
        # identical to train_and_save.py / app.py
        features['subdomain_count'] = max(0, len(parts) - 2)
        full_lower = url.lower()
        features['has_suspicious_keyword'] = 1 if any(k in full_lower for k in SUSPICIOUS_KEYWORDS) else 0
        features['uses_shortener'] = 1 if any(s in domain for s in URL_SHORTENERS) else 0
        features['uses_https'] = 1 if parsed.scheme.lower() == 'https' else 0
        features['has_hyphen_in_domain'] = 1 if '-' in domain else 0
        features['slash_count'] = url.count('/')
    except Exception as e:
        # لا نريد أن يكسر الفحص؛ نعيد dict فارغ لو حصل خطأ
        return {}
    return features

# ---------- تحميل الموديل والمحولة ----------
MODEL_FILE = 'phishing_model.joblib'
VECT_FILE = 'vectorizer.joblib'

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECT_FILE):
    print(f"❌ خطأ: تأكدي أن '{MODEL_FILE}' و'{VECT_FILE}' موجودين في نفس المجلد ({os.getcwd()})")
    sys.exit(1)

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECT_FILE)
print("✅ تم تحميل النموذج والمحولة بنجاح\n")

# ---------- اختبار على URL تجريبي ----------
test_url = "https://paypal-login-verification.com"  # غيّريه لو حابة تجرب URL آخر
features = extract_features(test_url)

print("🔎 الميزات المستخرجة:")
pprint(features)

if not features:
    print("❌ لم يتم استخراج أي ميزة من الرابط المدخل، تأكدي من صحة الرابط.")
    sys.exit(1)

# تحويل الميزات بواسطة ال-vectorizer
try:
    X = vectorizer.transform([features])
except Exception as e:
    print("❌ خطأ عند تحويل الميزات باستخدام vectorizer:")
    print(str(e))
    sys.exit(1)

# طباعة فئات الموديل (ترتيبها مهم ل interpret)
print("\n📚 فئات الموديل (model.classes_):", model.classes_)

# توقع واحتساب احتمالات
try:
    pred = model.predict(X)
except Exception as e:
    print("❌ خطأ أثناء prediction:")
    print(str(e))
    sys.exit(1)

proba = None
try:
    proba = model.predict_proba(X)
except Exception:
    proba = None

print("\n✅ النتيجة الخام (prediction):", pred)
if proba is not None:
    print("📈 الاحتمالات (predict_proba):")
    pprint(proba.tolist())
else:
    print("⚠️ الموديل لا يدعم predict_proba أو حدث خطأ أثناء حسابها.")

# ---------- تفسير النتيجة بطريقة بسيطة ----------
raw_label = pred[0]
# حساب ثقة بسيطة
confidence = 0.0
if proba is not None:
    try:
        classes = list(model.classes_)
        if raw_label in classes:
            idx = classes.index(raw_label)
            confidence = float(proba[0][idx] * 100)
        else:
            # fallback: أعلى احتمال
            confidence = float(max(proba[0]) * 100)
    except Exception:
        confidence = 0.0

label_str = str(raw_label).lower()
if 'phish' in label_str or label_str in ('1', 'true', 'yes', 'malicious'):
    verdict = "خطر (Phishing)"
else:
    verdict = "آمن (Legitimate)"

print(f"\n🎯 التفسير: {verdict}")
print(f"🔢 الملصق الخام: {raw_label}")
print(f"✨ الثقة (تقريبية): {confidence:.2f}%")
