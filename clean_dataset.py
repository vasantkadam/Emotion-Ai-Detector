import pandas as pd

print("🚀 Script started")

# Load dataset
df = pd.read_csv("data/emotion_data.csv")
print("📂 Dataset loaded. Rows:", len(df))

remove_phrases = [
    "today",
    "these days",
    "right now",
    "at the moment"
]

def clean_text(text):
    text = str(text).lower()
    for phrase in remove_phrases:
        text = text.replace(phrase, "")
    return " ".join(text.split())

df["text"] = df["text"].apply(clean_text)

df.to_csv("data/emotion_data_cleaned.csv", index=False)

print("✅ Dataset cleaned and saved successfully")
