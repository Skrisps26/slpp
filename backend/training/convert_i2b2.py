"""
Convert i2b2 2010 XML data to HuggingFace NER format.
Input: i2b2 XML files with <TAGGING> blocks
Output: JSON files with [tokens, ner_tags] format

Usage: python -m backend.training.convert_i2b2 --input data/i2b2/xml --output data/ner/i2b2_train
"""
import os
import re
import json
import argparse
import xml.etree.ElementTree as ET


def parse_i2b2_xml(xml_path: str) -> list:
    """Parse an i2b2 XML file and extract token-label pairs."""
    from collections import defaultdict

    LABEL_MAP = {
        "problem": "DIAGNOSIS",
        "treatment": "MEDICATION",
        "test": "VITAL",
    }

    tree = ET.parse(xml_path)
    root = tree.getroot()

    text_elem = root.find("TEXT")
    tagging_elem = root.find("TAGGING")

    if text_elem is None or tagging_elem is None:
        return []

    text = text_elem.text

    # Extract all tags with their positions
    tags = []
    for tag_elem in tagging_elem:
        tag_type = tag_elem.attrib.get("TYPE", "")
        start = int(tag_elem.attrib.get("start", 0))
        end = int(tag_elem.attrib.get("end", 0))
        entity_type = LABEL_MAP.get(tag_type, "SYMPTOM")
        tags.append((start, end, entity_type))

    # Sort tags by start position
    tags.sort()

    # Simple tokenizer
    tokens = []
    tok_positions = []
    for match in re.finditer(r'\S+|\s+', text):
        token = match.group()
        if token.strip():
            tokens.append(token)
            tok_positions.append((match.start(), match.end()))

    # Assign BIO labels
    labels = ["O"] * len(tokens)
    for tok_idx, (tok_start, tok_end) in enumerate(tok_positions):
        for tag_start, tag_end, entity_type in tags:
            if tok_start >= tag_start and tok_start < tag_end:
                prefix = "B"
                # Check if previous token is same entity
                if tok_idx > 0 and tok_positions[tok_idx - 1][1] > tag_start:
                    prefix = "I"
                labels[tok_idx] = f"{prefix}-{entity_type}"
                break

    return tokens, labels


def main():
    parser = argparse.ArgumentParser(description="Convert i2b2 XML to HuggingFace NER format")
    parser.add_argument("--input", required=True, help="Path to i2b2 XML directory")
    parser.add_argument("--output", required=True, help="Output directory for JSON files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_tokens = []
    all_tags = []
    label_vocab = {}
    label_idx = 0

    for filename in os.listdir(args.input):
        if not filename.endswith(".xml"):
            continue
        filepath = os.path.join(args.input, filename)
        try:
            tokens, labels = parse_i2b2_xml(filepath)
            if not tokens:
                continue

            numeric_labels = []
            for label in labels:
                if label not in label_vocab:
                    label_vocab[label] = label_idx
                    label_idx += 1
                numeric_labels.append(label_vocab[label])

            all_tokens.append(tokens)
            all_tags.append(numeric_labels)
            print(f"  Converted {filename}: {len(tokens)} tokens")
        except Exception as e:
            print(f"  Error converting {filename}: {e}")

    # Split into train/validation
    split_idx = int(len(all_tokens) * 0.8)

    train_data = {"tokens": all_tokens[:split_idx], "ner_tags": all_tags[:split_idx]}
    val_data = {"tokens": all_tokens[split_idx:], "ner_tags": all_tags[split_idx:]}

    with open(os.path.join(args.output, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(args.output, "validation.json"), "w") as f:
        json.dump(val_data, f, indent=2)

    # Save label mapping
    label_map_path = os.path.join(args.output, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_vocab, f, indent=2)

    print(f"\n[convert_i2b2] Conversion complete:")
    print(f"  Train: {len(train_data['tokens'])} examples")
    print(f"  Validation: {len(val_data['tokens'])} examples")
    print(f"  Labels ({len(label_vocab)}): {label_vocab}")


if __name__ == "__main__":
    main()
