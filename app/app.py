"""
FinRiskGuard — Gradio Demo UI

Runs as a standalone demo that calls the FastAPI backend.
Two tabs: Fraud Detection | Credit Scoring
"""

import os
import json
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────

def call_fraud_predict(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/api/v1/fraud/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def call_credit_predict(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/api/v1/credit/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def call_fraud_metadata() -> dict:
    r = requests.get(f"{API_URL}/api/v1/fraud/metadata", timeout=10)
    r.raise_for_status()
    return r.json()


def call_credit_metadata() -> dict:
    r = requests.get(f"{API_URL}/api/v1/credit/metadata", timeout=10)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# Fraud tab logic
# ─────────────────────────────────────────────

def predict_fraud(
    transaction_id, transaction_dt, transaction_amt,
    product_cd, card4, card6, p_emaildomain, r_emaildomain,
    device_type,
):
    payload = {
        "TransactionID" : int(transaction_id),
        "TransactionDT" : int(transaction_dt),
        "TransactionAmt": float(transaction_amt),
        "ProductCD"     : product_cd,
        "card4"         : card4,
        "card6"         : card6,
        "P_emaildomain" : p_emaildomain,
        "R_emaildomain" : r_emaildomain,
        "DeviceType"    : device_type,
    }
    try:
        result = call_fraud_predict(payload)
        prob   = result["fraud_probability"]
        label  = result["decision"]
        thr    = result["threshold"]

        color  = "🔴" if result["is_fraud"] else "🟢"
        output = (
            f"{color} **{label}**\n\n"
            f"Fraud Probability : `{prob:.4f}`\n"
            f"Threshold         : `{thr}`\n"
            f"Model             : `{result['model']}`"
        )
        gauge = round(prob * 100, 2)
        return output, gauge
    except Exception as e:
        return f"❌ Error: {e}", 0.0


# ─────────────────────────────────────────────
# Credit tab logic
# ─────────────────────────────────────────────

def predict_credit(
    sk_id_curr, amt_credit, amt_income, amt_annuity,
    days_birth, days_employed,
    ext1, ext2, ext3,
):
    payload = {
        "SK_ID_CURR"      : int(sk_id_curr),
        "AMT_CREDIT"      : float(amt_credit),
        "AMT_INCOME_TOTAL": float(amt_income),
        "AMT_ANNUITY"     : float(amt_annuity),
        "DAYS_BIRTH"      : int(days_birth),
        "DAYS_EMPLOYED"   : int(days_employed),
    }
    if ext1 is not None: payload["EXT_SOURCE_1"] = float(ext1)
    if ext2 is not None: payload["EXT_SOURCE_2"] = float(ext2)
    if ext3 is not None: payload["EXT_SOURCE_3"] = float(ext3)

    try:
        result = call_credit_predict(payload)
        prob   = result["default_probability"]
        label  = result["risk_label"]
        thr    = result["threshold"]

        color  = "🔴" if result["will_default"] else "🟢"
        output = (
            f"{color} **{label}**\n\n"
            f"Default Probability : `{prob:.4f}`\n"
            f"Threshold           : `{thr}`\n"
            f"Model               : `{result['model']}`"
        )
        gauge = round(prob * 100, 2)
        return output, gauge
    except Exception as e:
        return f"❌ Error: {e}", 0.0


# ─────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────

with gr.Blocks(title="FinRiskGuard Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🛡️ FinRiskGuard
    ### Financial Risk ML — Fraud Detection & Credit Default Scoring
    Two production-grade models | XGBoost · LightGBM · CatBoost · Stacking Ensemble
    """)

    with gr.Tabs():

        # ── FRAUD TAB ──────────────────────────────────────────────
        with gr.Tab("🔴 Fraud Detection"):
            gr.Markdown("### Score a Transaction — IEEE-CIS Dataset | AUC-ROC: 0.9258 | Threshold: 0.44")

            with gr.Row():
                with gr.Column():
                    txn_id   = gr.Number(label="TransactionID",  value=2987004, precision=0)
                    txn_dt   = gr.Number(label="TransactionDT (seconds from 2017-12-01)", value=86400, precision=0)
                    txn_amt  = gr.Number(label="TransactionAmt ($)", value=117.5)
                    prod_cd  = gr.Dropdown(["W", "H", "C", "S", "R"], label="ProductCD", value="W")
                    card4    = gr.Dropdown(["visa", "mastercard", "american express", "discover"],
                                           label="card4", value="visa")
                    card6    = gr.Dropdown(["debit", "credit", "debit or credit", "charge card"],
                                           label="card6", value="debit")
                    p_email  = gr.Textbox(label="P_emaildomain", value="gmail.com")
                    r_email  = gr.Textbox(label="R_emaildomain", value="gmail.com")
                    dev_type = gr.Dropdown(["desktop", "mobile"], label="DeviceType", value="desktop")
                    btn_f    = gr.Button("🔍 Predict Fraud", variant="primary")

                with gr.Column():
                    fraud_result = gr.Markdown(label="Result")
                    fraud_gauge  = gr.Number(label="Fraud Probability (%)")

            btn_f.click(
                predict_fraud,
                inputs=[txn_id, txn_dt, txn_amt, prod_cd, card4, card6, p_email, r_email, dev_type],
                outputs=[fraud_result, fraud_gauge],
            )

            with gr.Accordion("📊 Model Metadata", open=False):
                def load_fraud_meta():
                    try:
                        m = call_fraud_metadata()
                        return json.dumps(m, indent=2)
                    except Exception as e:
                        return str(e)
                meta_btn_f = gr.Button("Load Metadata")
                meta_out_f = gr.Code(language="json")
                meta_btn_f.click(load_fraud_meta, outputs=meta_out_f)

        # ── CREDIT TAB ─────────────────────────────────────────────
        with gr.Tab("🟠 Credit Scoring"):
            gr.Markdown("### Score a Loan Applicant — Home Credit Dataset | AUC-ROC: 0.7849 | Threshold: 0.50")

            with gr.Row():
                with gr.Column():
                    sk_id    = gr.Number(label="SK_ID_CURR", value=100002, precision=0)
                    amt_cred = gr.Number(label="AMT_CREDIT ($)", value=406597.5)
                    amt_inc  = gr.Number(label="AMT_INCOME_TOTAL ($)", value=202500.0)
                    amt_ann  = gr.Number(label="AMT_ANNUITY ($)", value=24700.5)
                    d_birth  = gr.Number(label="DAYS_BIRTH (negative)", value=-9461, precision=0)
                    d_emp    = gr.Number(label="DAYS_EMPLOYED (negative, or 365243 if anomaly)", value=-637, precision=0)
                    ext1     = gr.Number(label="EXT_SOURCE_1 (optional, 0–1)", value=None)
                    ext2     = gr.Number(label="EXT_SOURCE_2 (optional, 0–1)", value=0.651)
                    ext3     = gr.Number(label="EXT_SOURCE_3 (optional, 0–1)", value=0.493)
                    btn_c    = gr.Button("🔍 Predict Default Risk", variant="primary")

                with gr.Column():
                    credit_result = gr.Markdown(label="Result")
                    credit_gauge  = gr.Number(label="Default Probability (%)")

            btn_c.click(
                predict_credit,
                inputs=[sk_id, amt_cred, amt_inc, amt_ann, d_birth, d_emp, ext1, ext2, ext3],
                outputs=[credit_result, credit_gauge],
            )

            with gr.Accordion("📊 Model Metadata", open=False):
                def load_credit_meta():
                    try:
                        m = call_credit_metadata()
                        return json.dumps(m, indent=2)
                    except Exception as e:
                        return str(e)
                meta_btn_c = gr.Button("Load Metadata")
                meta_out_c = gr.Code(language="json")
                meta_btn_c.click(load_credit_meta, outputs=meta_out_c)

    gr.Markdown("""
    ---
    **FinRiskGuard** · FastAPI + Docker + AWS EC2 · XGBoost · LightGBM · CatBoost · Stacking · SHAP
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
