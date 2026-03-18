from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np

class DLDocumentClassifierTF:
    def __init__(self, model_path="distilbert-base-uncased", num_labels=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels
        )

    def predict(self, text, temperature=1.5):
        inputs = self.tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=512
        )

        logits = self.model(inputs).logits

        scaled_logits = logits / temperature
        probs = tf.nn.softmax(scaled_logits, axis=1).numpy()[0]

        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]

        return pred_idx, float(confidence)