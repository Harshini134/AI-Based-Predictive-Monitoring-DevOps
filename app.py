import gradio as gr
import pandas as pd
import joblib

model = joblib.load("devops_risk_model.pkl")

def predict_server_risk(cpu, memory, disk, response_time):
    data = pd.DataFrame([{
        "CPU_Usage": cpu,
        "Memory_Usage": memory,
        "Disk_Usage": disk,
        "Response_Time": response_time
    }])

    risk = model.predict(data)[0]
    return f"Predicted Server Risk: {risk}"

with gr.Blocks(title="DevOps Risk Monitoring") as demo:
    gr.Markdown("# AI-Based Predictive Monitoring for DevOps Infrastructure")
    gr.Markdown("Enter server metrics to estimate risk.")

    cpu = gr.Slider(0, 100, value=50, label="CPU Usage (%)")
    memory = gr.Slider(0, 100, value=50, label="Memory Usage (%)")
    disk = gr.Slider(0, 100, value=50, label="Disk Usage (%)")
    response = gr.Slider(50, 1000, value=300, label="Response Time (ms)")

    button = gr.Button("Predict Server Risk")
    output = gr.Textbox(label="Prediction")

    button.click(
        predict_server_risk,
        inputs=[cpu, memory, disk, response],
        outputs=output
    )

demo.launch()
