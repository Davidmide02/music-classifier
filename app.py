from transformers import pipeline
import gradio as gr

pipe = pipeline(
    "audio-classification", model="Davidmide02/distilhubert-finetuned-gtzan"
)

def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs

demo = gr.Interface(
    fn=classify_audio, inputs=gr.Audio(type="filepath"), outputs=gr.Label()
)
demo.launch(debug=True)