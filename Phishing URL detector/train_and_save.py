# train_and_save.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import joblib
import re
import ipaddress
from urllib.parse import urlparse
import os
import sys

# ---------- ثوابت ----------
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'account', 'update', 'secure',
    'banking', 'ebay', 'paypal', 'amazon', 'facebook', 'google', 'apple', 'microsoft'
]
URL_SHORTENERS = [
    'bit.ly', 't.co', 'tinyurl.com', 'goo.gl', 'ow.ly', 'buff.ly', 't.ly'
]

# ---------- دالة استخراج الميزات ----------
def extract_features(url):
    features = {}
    if not isinstance(url, str) or url.strip() == '':
        return {}
    url = url.strip()
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
        features['subdomain_count'] = max(0, len(parts) - 2)
        full_lower = url.lower()
        features['has_suspicious_keyword'] = 1 if any(k in full_lower for k in SUSPICIOUS_KEYWORDS) else 0
        features['uses_shortener'] = 1 if any(s in domain for s in URL_SHORTENERS) else 0
        features['uses_https'] = 1 if parsed.scheme.lower() == 'https' else 0
        features['has_hyphen_in_domain'] = 1 if '-' in domain else 0
        features['slash_count'] = url.count('/')
    except Exception:
        return {}
    return features

# ---------- مساعدة لإيجاد الأعمدة ----------
def find_column_name(cols, candidates):
    cols_low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_low:
            return cols_low[cand.lower()]
    return None

# ---------- main ----------
CSV = 'malicious_phish.csv'
MODEL_OUT = 'phishing_model.joblib'
VECT_OUT = 'vectorizer.joblib'

print("بدء عملية تدريب النموذج...")

if not os.path.exists(CSV):
    print(f"❌ خطأ: ملف البيانات '{CSV}' غير موجود في المجلد الحالي ({os.getcwd()})")
    sys.exit(1)

df = pd.read_csv(CSV)
print(f"✅ تم تحميل الملف: {CSV} — عدد الصفوف: {len(df)}")

# تنظيف أسماء الأعمدة
df.columns = df.columns.str.strip()
cols = list(df.columns)
print("📊 الأعمدة المتاحة:", cols)

# تحديد أعمدة URL و label
possible_url = ['url', 'urls', 'link', 'links', 'website', 'domain', 'address', 'uri']
possible_label = ['label', 'labels', 'target', 'is_phishing', 'phishing', 'class', 'type', 'y']

url_col = find_column_name(cols, possible_url)
label_col = find_column_name(cols, possible_label)

if url_col is None:
    print("❌ لم أجد عمود URL مناسب. تأكدي أن فيه عمود اسمه مثل 'url' أو 'link'.")
    sys.exit(1)
if label_col is None:
    print("❌ لم أجد عمود label مناسب. تأكدي أن فيه عمود اسمه مثل 'label' أو 'type'.")
    sys.exit(1)

print(f"🧩 استخدمنا '{url_col}' كعمود URL و'{label_col}' كعمود label.")

# تجهيز القيم في العمود label
df[label_col] = df[label_col].astype(str).str.lower().str.strip()
if df[label_col].nunique() > 2:
    print("⚠️ عمود التصنيف يحتوي على أكثر من فئتين. سيتم استخدامه كما هو.")
else:
    # تحويل phishing → 1 ، benign → 0
    mapping = {
        'phishing': 1,
        'malicious': 1,
        'bad': 1,
        'legitimate': 0,
        'benign': 0,
        'good': 0
    }
    df[label_col] = df[label_col].map(lambda x: mapping.get(x, x))

# استخراج الميزات
features_list = []
labels = []
skipped = 0

for _, row in df.iterrows():
    url_val = row.get(url_col, '')
    feats = extract_features(url_val)
    if feats:
        features_list.append(feats)
        labels.append(row.get(label_col))
    else:
        skipped += 1

print(f"✅ تم استخراج ميزات من {len(features_list)} روابط. تم تخطي {skipped} صف/صفوف.")

if len(features_list) == 0:
    print("❌ لم يتم استخراج أي ميزات من الروابط. تأكدي من محتوى عمود URLs.")
    sys.exit(1)

# تحويل للـ matrix
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(features_list)
y = labels

# تقسيم وتدريب
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
)

print("🧠 جاري تدريب نموذج RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("🎯 اكتمل التدريب بنجاح!")

# حفظ النموذج والمحولات
joblib.dump(model, MODEL_OUT)
joblib.dump(vec, VECT_OUT)

print("-" * 40)
print(f"✅ تم حفظ النموذج في: '{MODEL_OUT}'")
print(f"✅ تم حفظ المحول في: '{VECT_OUT}'")
print("-" * 40)
print("✅ جاهز للاستخدام في API أو inference.")