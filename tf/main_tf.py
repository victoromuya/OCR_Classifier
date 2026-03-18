from file_loader import extract_text
from preprocess import clean_text
from extractor import extract_entities, extract_amount
from dl_classifier_tf import DLDocumentClassifierTF
from label_map import label_map

classifier = DLDocumentClassifierTF(model_path="tf_model")

def process_document(file_path):
    raw_text = extract_text(file_path)

    cleaned = clean_text(raw_text)

    pred_idx, confidence = classifier.predict(cleaned)
    doc_type = label_map[pred_idx]

    entities = extract_entities(raw_text)
    amount = extract_amount(raw_text)

    return {
        "document_type": doc_type,
        "confidence": confidence,
        "entities": entities,
        "amount": amount
    }


if __name__ == "__main__":
    file_path = "test_files/sample.pdf"

    result = process_document(file_path)

    print("\n📄 Result:")
    print(result)