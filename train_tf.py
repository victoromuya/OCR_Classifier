import os
from file_loader import extract_text
from preprocess import clean_text, tokenize
from label_map import reverse_label_map
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import Dataset
import tensorflow as tf

def load_dataset(data_dir, limit_per_class=5):
    texts = []
    labels = []

    for label in os.listdir(data_dir):
        folder = os.path.join(data_dir, label)

        if not os.path.isdir(folder):
            continue

        files = sorted(os.listdir(folder))[:limit_per_class]

        for file in files:
            path = os.path.join(folder, file)

            try:
                text = extract_text(path)
                processed = tokenize(clean_text(text))

                texts.append(processed)
                labels.append(reverse_label_map[label])

            except Exception as e:
                print(f"Skipping {file}: {e}")

    return texts, labels


# Load data
texts, labels = load_dataset("data/")

dataset = Dataset.from_dict({
    "text": texts,
    "label": labels
})

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize_fn, batched=True)

tf_dataset = dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    batch_size=8,
    shuffle=True
)

model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(tf_dataset, epochs=3)

# Save model
model.save_pretrained("tf_model")
tokenizer.save_pretrained("tf_model")

print("✅ Model trained and saved!")