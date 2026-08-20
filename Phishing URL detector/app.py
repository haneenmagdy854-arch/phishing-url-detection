# app.py (updated)
from flask import Flask, request, jsonify, render_template
import joblib
import re
import ipaddress
from urllib.parse import urlparse
import os
import sys
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# ---------- ثوابت و extract_features ----------
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

# ---------- تحميل النموذج و vectorizer ----------
MODEL_FILE = 'phishing_model.joblib'
VECT_FILE = 'vectorizer.joblib'

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECT_FILE):
    print(f"خطأ: تأكد من وجود '{MODEL_FILE}' و'{VECT_FILE}' في مجلد المشروع ({os.getcwd()})")
    sys.exit(1)

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECT_FILE)
print("✅ النموذج والمحولة تم تحميلهم بنجاح")

# ---------- مساعدة تفسير ----------
def interpret_prediction(prediction, proba=None):
    raw_label = prediction[0]
    confidence = 0.0
    if proba is not None:
        try:
            classes = list(model.classes_)
            if raw_label in classes:
                idx = classes.index(raw_label)
                confidence = float(proba[0][idx] * 100)
            else:
                try:
                    idx = classes.index(int(raw_label))
                    confidence = float(proba[0][idx] * 100)
                except Exception:
                    confidence = float(max(proba[0]) * 100)
        except Exception:
            confidence = 0.0

    label_str = str(raw_label).lower()
    if 'phish' in label_str or label_str in ('1', 'true', 'yes', 'malicious'):
        return "خطر (Phishing)", round(confidence, 2), "#dc3545", raw_label
    else:
        return "آمن (Legitimate)", round(confidence, 2), "#28a745", raw_label

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/info', methods=['GET'])
def api_info():
    try:
        classes = [str(c) for c in model.classes_]
    except Exception:
        classes = []
    return jsonify({
        "model_file": MODEL_FILE,
        "vectorizer_file": VECT_FILE,
        "classes": classes
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json(force=True, silent=True)
    if not data or 'url' not in data:
        return jsonify({"error": "يرجى إرسال JSON يحتوي على المفتاح 'url'"}), 400

    url_to_test = data['url']
    features = extract_features(url_to_test)
    if not features:
        return jsonify({"url": url_to_test, "result": "Invalid Link", "confidence": 0}), 400

    try:
        transformed = vectorizer.transform([features])
    except Exception as e:
        return jsonify({"url": url_to_test, "result": "Feature Error", "error": str(e)}), 500

    try:
        prediction = model.predict(transformed)
    except Exception as e:
        return jsonify({"url": url_to_test, "result": "Model Prediction Error", "error": str(e)}), 500

    proba = None
    try:
        proba = model.predict_proba(transformed)
    except Exception:
        proba = None

    verdict, confidence, color, raw_label = interpret_prediction(prediction, proba)

    return jsonify({
        "url": url_to_test,
        "result": verdict,
        "raw_label": str(raw_label),
        "confidence": confidence
    })

@app.route('/api/debug', methods=['POST'])
def api_debug():
    """
    يرجع: features, prediction, probabilities (if available),
    feature_names, feature_importances (if available), feature_contributions (importance * value)
    """
    data = request.get_json(force=True, silent=True)
    if not data or 'url' not in data:
        return jsonify({"error": "يرجى إرسال JSON يحتوي على المفتاح 'url'"}), 400

    url_to_test = data['url']
    feats = extract_features(url_to_test)
    if not feats:
        return jsonify({"url": url_to_test, "error": "Invalid Link", "features": {}}), 400

    # vectorize
    try:
        X = vectorizer.transform([feats])
    except Exception as e:
        return jsonify({"url": url_to_test, "error": "Vectorize Error", "detail": str(e)}), 500

    # predict
    try:
        pred = model.predict(X)
    except Exception as e:
        return jsonify({"url": url_to_test, "error": "Prediction Error", "detail": str(e)}), 500

    proba = None
    try:
        proba = model.predict_proba(X).tolist()[0]
    except Exception:
        proba = None

    resp = {
        "url": url_to_test,
        "features": feats,
        "prediction": str(pred[0]),
        "probabilities": proba
    }

    # try to extract feature names from vectorizer
    feature_names = None
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        try:
            feature_names = vectorizer.feature_names_
        except Exception:
            feature_names = None

    # feature importances & contributions
    try:
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            importances = np.array(model.feature_importances_)
            x_arr = X.toarray().ravel()
            contributions = (importances * x_arr).tolist()
            resp['feature_names'] = list(feature_names)
            resp['feature_importances'] = importances.tolist()
            resp['feature_contributions'] = contributions
        else:
            resp['feature_names'] = list(feature_names) if feature_names is not None else None
    except Exception as e:
        resp['contrib_error'] = str(e)

    return jsonify(resp)

# optional: support form POST if someone posts form to /analyze (kept for backward)
@app.route('/analyze', methods=['POST'])
def analyze_form():
    url_to_test = request.form.get('url', '')
    features = extract_features(url_to_test)
    if not features:
        return render_template('index.html', result="Invalid Link", url=url_to_test, confidence_color="#ffc107")
    transformed = vectorizer.transform([features])
    prediction = model.predict(transformed)
    proba = None
    try:
        proba = model.predict_proba(transformed)
    except Exception:
        proba = None
    verdict, confidence, color, _ = interpret_prediction(prediction, proba)
    result_text = f"{verdict} (Confidence {confidence:.2f}%)"
    return render_template('index.html', result=result_text, url=url_to_test, confidence_color=color)

if __name__ == '__main__':
    # Use use_reloader=False to avoid duplicate processes and noisy errors on Ctrl+C
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)