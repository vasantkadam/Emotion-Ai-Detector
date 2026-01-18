from flask import Flask, render_template, request
import torch
import pickle
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from collections import Counter

app = Flask(__name__)

model_path = os.path.join(os.getcwd(), "model", "bert_emotion_model")

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

with open(os.path.join(model_path, "labels.pkl"), "rb") as f:
    labels = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

history = []

def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return labels[pred], probs[0][pred].item()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form["text"]
        emotion, confidence = predict_emotion(text)

        result = {
            "text": text,
            "emotion": emotion,
            "confidence": f"{confidence*100:.2f}%"
        }

        history.append(result)

        if len(history) > 5:
            history.pop(0)

    # Chart data
    counts = Counter([h["emotion"] for h in history])
    chart_data = {
        "labels": list(counts.keys()),
        "counts": list(counts.values())
    }

    return render_template("index.html", result=result, history=history, chart_data=chart_data)

if __name__ == "__main__":
    app.run(debug=True)
