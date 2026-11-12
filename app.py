import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session

import nids_common as nc

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'network-security-project-key')

HISTORY_LIMIT = 15
HIGH_RISK = ('High Risk', 'Critical Risk')

# Both algorithms are trained by train_compare.py on the same data, the same 5-class
# mapping and the same 11 features in the same order (nids_common.FEATURES).
MODEL_FILES = {'lr': 'model_lr.pkl', 'rf': 'model_rf.pkl'}
MODEL_NAMES = {'lr': 'Logistic Regression', 'rf': 'Random Forest'}
DEFAULT_ALGORITHM = 'lr'

# Input spec for the 10 form fields. The model's 11 features are derived from
# them: 'flag' becomes flag_S0 + flag_SF. kind: int / rate (0-1) / choice.
FIELDS = {
    'count':                       dict(kind='int', min=0, max=511),
    'dst_host_diff_srv_rate':      dict(kind='rate'),
    'dst_host_same_src_port_rate': dict(kind='rate'),
    'dst_host_same_srv_rate':      dict(kind='rate'),
    'dst_host_srv_count':          dict(kind='int', min=0, max=255),
    'flag':                        dict(kind='choice', choices={'0': 'Other', '1': 'S0', '2': 'SF'}),
    'logged_in':                   dict(kind='choice', choices={'0': 'No', '1': 'Yes'}),
    'same_srv_rate':               dict(kind='rate'),
    'serror_rate':                 dict(kind='rate'),
    'service_http':                dict(kind='choice', choices={'0': 'No', '1': 'Yes'}),
}

# ---- load both models once; a missing/unreadable model is reported, not fatal ----
models, model_errors = {}, {}
for _key, _file in MODEL_FILES.items():
    try:
        _m = joblib.load(nc.MODELS_DIR / _file)
        # Safety checks: identical feature order and the exact 5-class output for every model.
        if list(_m.feature_names_in_) != nc.FEATURES:
            raise ValueError('feature order does not match nids_common.FEATURES')
        if [int(c) for c in _m.classes_] != nc.CLASS_IDS:
            raise ValueError('model classes are not the five classes 0-4')
        models[_key] = _m
    except FileNotFoundError:
        model_errors[_key] = f'{MODEL_NAMES[_key]} model file models/{_file} not found. Run: python train_compare.py'
    except Exception as exc:  # e.g. scikit-learn version mismatch
        model_errors[_key] = f'Could not load {MODEL_NAMES[_key]} (models/{_file}): {exc}. Re-run: python train_compare.py'


def validate(data):
    """Validate raw form/JSON values. Returns (clean_values, errors)."""
    clean, errors = {}, {}
    for name, spec in FIELDS.items():
        raw = str(data.get(name, '')).strip()
        if raw == '':
            errors[name] = 'Required.'
            continue
        if spec['kind'] == 'choice':
            if raw not in spec['choices']:
                errors[name] = 'Choose one of: ' + ', '.join(spec['choices'])
            else:
                clean[name] = raw
            continue
        try:
            num = float(raw)
        except ValueError:
            errors[name] = 'Enter a number.'
            continue
        lo, hi = (spec['min'], spec['max']) if spec['kind'] == 'int' else (0.0, 1.0)
        if not np.isfinite(num) or not lo <= num <= hi:
            errors[name] = f'Must be between {lo} and {hi}.'
        elif spec['kind'] == 'int' and num != int(num):
            errors[name] = 'Must be a whole number.'
        else:
            clean[name] = int(num) if spec['kind'] == 'int' else num
    return clean, errors


def choose_algorithm(data):
    """Return (key, error). The algorithm must be a known, successfully loaded model."""
    key = str(data.get('algorithm', DEFAULT_ALGORITHM)).strip() or DEFAULT_ALGORITHM
    if key not in MODEL_FILES:
        return None, 'Choose Logistic Regression or Random Forest.'
    if key not in models:
        return None, model_errors.get(key, f'{MODEL_NAMES[key]} is not available.')
    return key, None


def to_features(v):
    """Validated inputs -> one-row DataFrame in the exact training feature order.
    The same frame is given to either algorithm. No scaling here: the Logistic
    Regression pipeline scales internally, and Random Forest needs no scaling."""
    flag_s0, flag_sf = nc.encode_flag({'1': 'S0', '2': 'SF'}.get(v['flag'], 'OTHER'))
    row = {
        'count': v['count'],
        'dst_host_diff_srv_rate': v['dst_host_diff_srv_rate'],
        'dst_host_same_src_port_rate': v['dst_host_same_src_port_rate'],
        'dst_host_same_srv_rate': v['dst_host_same_srv_rate'],
        'dst_host_srv_count': v['dst_host_srv_count'],
        'flag_S0': flag_s0,
        'flag_SF': flag_sf,
        'logged_in': int(v['logged_in']),
        'same_srv_rate': v['same_srv_rate'],
        'serror_rate': v['serror_rate'],
        'service_http': int(v['service_http']),
    }
    return pd.DataFrame([row], columns=nc.FEATURES)


def classify(key, X):
    model = models[key]
    proba = model.predict_proba(X)[0]
    best = int(np.argmax(proba))
    class_id = int(model.classes_[best])
    return nc.CLASS_NAMES[class_id], nc.RISK_LEVELS[class_id], round(float(proba[best]) * 100, 2)


def get_stats():
    """Running totals for this browser session (kept separately so they are not
    limited to the 15 rows shown in the history table)."""
    stats = session.get('stats')
    if stats is None:  # session created before the dashboard existed: rebuild from history
        hist = session.get('history', [])
        stats = {'total': len(hist),
                 'normal': sum(h['attack'] == 'Normal' for h in hist),
                 'high': sum(h['risk'] in HIGH_RISK for h in hist)}
    return stats


def summary():
    stats = get_stats()
    return {'total': stats['total'], 'normal': stats['normal'],
            'attacks': stats['total'] - stats['normal'], 'high': stats['high']}


def page(**ctx):
    ctx.setdefault('history', session.get('history', []))
    ctx.setdefault('summary', summary())
    ctx.setdefault('model_info', {'features': len(nc.FEATURES), 'classes': len(nc.CLASS_NAMES)})
    ctx.setdefault('algorithms', [(k, MODEL_NAMES[k], k in models) for k in MODEL_FILES])
    ctx.setdefault('selected_algorithm', DEFAULT_ALGORITHM)
    ctx.setdefault('errors', {})
    ctx.setdefault('values', {})
    ctx.setdefault('model_error', ' | '.join(model_errors.values()) or None)
    return render_template('index.html', **ctx)


@app.route('/')
def home():
    return page()


@app.route('/predict', methods=['POST'])
def predict():
    selected = request.form.get('algorithm', DEFAULT_ALGORITHM)
    key, algo_error = choose_algorithm(request.form)
    values, errors = validate(request.form)
    if algo_error:
        errors['_'] = algo_error
    if errors:
        return page(errors=errors, values=request.form, selected_algorithm=selected), (503 if not models else 400)

    try:
        output, risk_level, confidence = classify(key, to_features(values))
    except Exception:  # never crash the page on an unexpected model error
        app.logger.exception('Prediction failed')
        return page(errors={'_': 'Prediction failed unexpectedly. Please check the inputs and try again.'},
                    values=request.form, selected_algorithm=selected), 500

    # Update the running totals first (get_stats may rebuild from the old history),
    # then add this prediction to the history table.
    stats = dict(get_stats())
    stats['total'] += 1
    stats['normal'] += output == 'Normal'
    stats['high'] += risk_level in HIGH_RISK
    session['stats'] = stats

    history = session.get('history', [])
    history.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'algorithm': MODEL_NAMES[key],
                    'attack': output, 'risk': risk_level, 'confidence': confidence})
    session['history'] = history[-HISTORY_LIMIT:]

    return page(output=output, risk_level=risk_level, confidence=confidence, algorithm_name=MODEL_NAMES[key],
                risk_css=risk_level.split()[0].lower(), is_attack=output != 'Normal',
                values=request.form, history=session['history'], selected_algorithm=key)


@app.route('/results', methods=['POST'])
def results():
    """JSON API: send the 10 form fields (see FIELDS) plus optional "algorithm": "lr" or "rf"."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Send a JSON object.'), 400
    key, algo_error = choose_algorithm(data)
    if algo_error:
        return jsonify(error=algo_error), (503 if not models else 400)
    values, errors = validate(data)
    if errors:
        return jsonify(errors=errors), 400
    try:
        output, risk_level, confidence = classify(key, to_features(values))
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify(error='Prediction failed unexpectedly.'), 500
    return jsonify(algorithm=MODEL_NAMES[key], attack=output, risk=risk_level, confidence=confidence)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
