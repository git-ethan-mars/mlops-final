import json
import os
import random

import mlflow
import pandas as pd
import pycaret.classification
import pycaret.datasets
import sklearn
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.exceptions import InternalServerError

from my_logger import logger

experiment_name = "SalaryExperiment"
model_name = "SalaryPredictor"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow_client = mlflow.MlflowClient()

app = Flask(__name__)

ab_config = {
    "production": 0.7,
    "staging": 0.3
}

train_data = pd.read_csv('./data/adult.csv')

try:
    production_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    production_model_version = mlflow_client.get_latest_versions(model_name, ['Production'])[0].version
except mlflow.exceptions.MlflowException:
    production_model = None
    production_model_version = None
try:
    staging_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Staging")
    staging_model_version = mlflow_client.get_latest_versions(model_name, ['Staging'])[0].version
except mlflow.exceptions.MlflowException:
    staging_model = None
    staging_model_version = None


@app.post("/api/train")
def train_model():
    global production_model, production_model_version, staging_model, staging_model_version
    mlflow.end_run()
    with mlflow.start_run():
        pycaret.classification.setup(train_data, target='salary', session_id=123, log_experiment=False,
                                     experiment_name=experiment_name, n_jobs=1)
        best_model = pycaret.classification.compare_models(include=['rf', 'lightgbm'])
        final_model = pycaret.classification.finalize_model(best_model)
        pycaret.classification.predict_model(final_model)
        test_metrics = pycaret.classification.pull()

        test_f1 = test_metrics['F1'][0]
        test_accuracy = test_metrics['Accuracy'][0]
        test_auc = test_metrics['AUC'][0]
        mlflow.log_metric('f1', test_f1)
        mlflow.log_metric('accuracy', test_accuracy)
        mlflow.log_metric('auc', test_auc)

        X_sample = train_data.drop('salary', axis=1).head(5)
        y_sample = train_data['salary'].head(5)
        signature = mlflow.models.infer_signature(X_sample, y_sample)
        registered_model = mlflow.sklearn.log_model(sk_model=final_model, registered_model_name=model_name,
                                                    signature=signature, artifact_path='pipeline')
    model_version = registered_model.registered_model_version
    if model_version == '1':
        _promote_to_production(model_version)
        production_model = final_model
        production_model_version = model_version
    else:
        _promote_to_stage(model_version)
        staging_model = final_model
        staging_model_version = model_version

    return {
        "status": "success",
        "test_metrics": {
            "f1": test_f1,
            "accuracy": test_accuracy,
            "auc": test_auc
        }
    }


@app.post("/api/ab/config")
def update_ab_config():
    global ab_config

    weights = request.json
    sum_weights = weights['production'] + weights['staging']

    ab_config['staging'] = weights['staging'] / sum_weights
    ab_config['production'] = weights['production'] / sum_weights

    return ab_config, 200


@app.get("/api/ab/config")
def get_ab_config():
    return ab_config, 200


def parse_logs():
    log_file = 'logs/predictions.log'
    with open(log_file, 'r') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        json_start_index = line.find('{')
        if json_start_index != - 1:
            record = json.loads(line[json_start_index:])
            records.append(record)

    return pd.DataFrame(records)


def calculate_business_metrics(labeled_df):
    results = {}
    for stage in ['staging', 'production']:
        subset = labeled_df[labeled_df['stage'] == stage]

        if len(subset) == 0:
            results[stage] = {
                'accuracy': 0,
                'recall': 0,
                'f1_score': 0,
                'count': 0,
                'version': None
            }
        else:
            y_true = subset['target']

            if stage == 'staging' and staging_model_version != subset['version'].iloc[0]:
                y_pred = staging_model.predict(pd.DataFrame(subset['features']))
            else:
                y_pred = subset['prediction']
            acc = sklearn.metrics.accuracy_score(y_true, y_pred)
            rec = sklearn.metrics.recall_score(y_true, y_pred, zero_division=0, pos_label='>50K')
            f1 = sklearn.metrics.f1_score(y_true, y_pred, zero_division=0, pos_label='>50K')
            results[stage] = {
                'accuracy': acc,
                'recall': rec,
                'f1_score': f1,
                'count': len(y_pred),
                'version': subset['version'].iloc[0]
            }

    return results


@app.get("/api/ab/report")
def analyze():
    global production_model, production_model_version, staging_model, staging_model_version
    df = parse_logs()

    if df.empty:
        return "No logs found.", 200

    labeled_df = df.dropna(subset=['target'])

    metrics = calculate_business_metrics(labeled_df)

    report = "# A/B Test Results\n\n"
    report += "| Group | Accuracy | Recall | F1-Score | Count |\n"
    report += "|-------|----------|--------|----------|-------|\n"

    for group, m in metrics.items():
        report += f"| {group} | {m['accuracy']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['count']} |\n"

    if (metrics['staging']['accuracy'] > metrics['production']['accuracy']
            and metrics['staging']['recall'] > metrics['production']['recall']):
        conclusion = "+++ New model (B) performs better. Promote to Production."
        _promote_to_production(metrics['staging']['version'])
        production_model = staging_model
        production_model = staging_model_version
        staging_model = None
        staging_model_version = None
    else:
        conclusion = "--- Keep current model (A)."

    report += f"\n## === Conclusion\n{conclusion}\n"

    # Сохранение отчёта
    report_path = "logs/ab_test_summary.md"
    with open(report_path, 'w') as f:
        f.write(report)
    return report, 200


@app.get('/api/psi')
def get_psi():
    reference = train_data
    jsons = parse_logs()['features']
    numeric_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    psi_by_feature = {}

    for col in numeric_cols:
        values = []
        for j in jsons:
            values.append(j[col])
        current_col = pd.concat([pd.Series(values), reference[col]])
        psi_by_feature[col] = calculate_psi(reference[col], current_col)

    max_psi = max(psi_by_feature.values()) if psi_by_feature else 0

    result = {'psi' : max_psi}

    return jsonify(result), 200

def calculate_psi(expected, actual, bucket_type = 'quantile', n_buckets = 10):
    def _calculate_bin_percentages(data, breakpoints):
        buckets = np.digitize(data, breakpoints, right=True)
        counts = np.bincount(buckets, minlength=len(breakpoints) + 1)
        percentages = counts / len(data)
        percentages = np.where(percentages == 0, 1e-4, percentages)
        return percentages

    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    if bucket_type == 'quantile':
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_buckets + 1)[1:-1])
        breakpoints = np.unique(breakpoints)  # Убираем дубликаты
    else:
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        breakpoints = np.linspace(min_val, max_val, n_buckets + 1)[1:-1]
    expected_perc = _calculate_bin_percentages(expected, breakpoints)
    actual_perc = _calculate_bin_percentages(actual, breakpoints)
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return psi


@app.post("/api/predict")
def predict():
    target = None
    if 'salary' in request.json:
        train_data.loc[len(train_data)] = request.json
        target = request.json['salary']
        del request.json['salary']
    df = pd.DataFrame([request.json])

    rand = random.random()

    if not production_model:
        raise InternalServerError(f"No model available for Production.")
    if not staging_model or rand > ab_config['production']:
        model, stage, version = production_model, 'production', production_model_version
    else:
        model, stage, version = staging_model, 'staging', staging_model_version

    prediction = model.predict(df)[0]
    features = df.to_dict(orient='records')[0]

    result = {
        'stage': stage,
        'version': version,
        'features': features,
        'prediction': prediction,
        'target': target
    }

    json_str = json.dumps(result)

    logger.info(json_str)

    return json_str, 200


def _promote_to_production(version):
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")

    for v in versions:
        if v.current_stage == "Production":
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )

    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )


def _promote_to_stage(version):
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")

    for v in versions:
        if v.current_stage == "Staging":
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )
    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
