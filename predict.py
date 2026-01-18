import torch
import pickle
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_path = os.path.join(os.getcwd(), "model", "bert_emotion_model")

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

with open(os.path.join(model_path, "labels.pkl"), "rb") as f:
    labels = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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

if __name__ == "__main__":
    print("=== AI Emotion Detection System ===")
    text = input("Enter text: ")

    emotion, confidence = predict_emotion(text)
    print(f"😊 Detected Emotion: {emotion}")
    print(f"🔍 Confidence: {confidence*100:.2f}%")
