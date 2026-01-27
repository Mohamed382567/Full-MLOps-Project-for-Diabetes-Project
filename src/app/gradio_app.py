import os
import requests
import gradio as gr

# --- SYSTEM CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/predict")

def get_prediction(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    """
    Inference client logic. 
    Handles data casting and UI state management.
    """
    # 1. Defensive Data Casting: Handle NoneType/Empty inputs to prevent crashes
    # Using 'or 0' ensures numerical stability if fields are left blank
    try:
        payload = {
            "Pregnancies": int(pregnancies or 0),
            "Glucose": float(glucose or 0),
            "BloodPressure": float(bp or 0),
            "SkinThickness": float(skin or 0),
            "Insulin": float(insulin or 0),
            "BMI": float(bmi or 0),
            "DiabetesPedigreeFunction": float(dpf or 0),
            "Age": int(age or 0)
        }
    except ValueError as e:
        return f"<div style='color: red;'>❌ <b>Input Error:</b> Invalid numeric value.</div>"

    try:
        # 2. API Communication
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        
        # 3. Data Extraction
        # NOTE: Using 'confidence' key to match the FastAPI response
        data = response.json()
        conf = data.get("confidence", 0)
        prediction = data.get("prediction", 0)
        
        # 4. Professional UI Rendering (HTML/CSS)
        # Applying color-coded feedback based on diagnosis
        is_diabetic = (prediction == 1)
        bg_color = "#ffebee" if is_diabetic else "#e8f5e9"
        text_color = "#c62828" if is_diabetic else "#2e7d32"
        icon = "⚠️" if is_diabetic else "✅"
        label = "DIABETIC (High Risk)" if is_diabetic else "NON-DIABETIC (Low Risk)"

        result_html = f"""
        <div style='text-align: center; padding: 25px; border-radius: 15px; background-color: {bg_color}; border: 2px solid {text_color}'>
            <h2 style='color: {text_color}; margin-top: 0;'>{icon} {label}</h2>
            <hr style='border: 0; border-top: 1px solid {text_color}; opacity: 0.3'>
            <p style='font-size: 0.9em; color: #555;'><i>Sensitivity Mode: High-Recall (Threshold: 0.40)</i></p>
        </div>
        """
        return result_html

    except requests.exceptions.ConnectionError:
        return "<div style='color: red; text-align: center;'>❌ <b>Service Offline:</b> Connection to API failed.</div>"
    except Exception as e:
        return f"<div style='color: red; text-align: center;'>❌ <b>Runtime Error:</b> {str(e)}</div>"

# --- UI ARCHITECTURE ---
def build_ui():
    # Applying the 'Soft' theme for a modern look
    with gr.Blocks(title="Diabetes AI Lab", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏥 Clinical Diabetes Prediction System")
        gr.Markdown("Architecture: **Distributed Microservices** | Engine: **Docker**")
        
        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                gr.Markdown("### Patient Metrics")
                preg = gr.Number(label="Pregnancies", value=0, minimum=0)
                gluc = gr.Number(label="Glucose (mg/dL)", value=120, minimum=0)
                bp = gr.Number(label="Blood Pressure (mm Hg)", value=70, minimum=0)
                skin = gr.Number(label="Skin Thickness (mm)", value=20, minimum=0)
                ins = gr.Number(label="Insulin (mu U/ml)", value=80, minimum=0)
                bmi = gr.Number(label="BMI (Weight in kg/(m)^2)", value=26.0, minimum=0)
                dpf = gr.Number(label="Pedigree Function", value=0.5, minimum=0)
                age = gr.Number(label="Age", value=30, minimum=1, maximum=120)
                
                btn = gr.Button("Analyze Patient Record", variant="primary")
            
            # Output Column
            with gr.Column(scale=1):
                gr.Markdown("### Diagnostic Intelligence")
                output = gr.HTML(label="Output Report")

        # Event Binding
        btn.click(
            fn=get_prediction, 
            inputs=[preg, gluc, bp, skin, ins, bmi, dpf, age], 
            outputs=output
        )

    return demo

if __name__ == "__main__":
    app = build_ui()
    port = int(os.environ.get("PORT", 8000))
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
