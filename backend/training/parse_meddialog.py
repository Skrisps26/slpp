"""
Convert raw MedDialog JSON into labeled Dialogue Act training data.
Auto-labels sentences using semantic heuristics.
"""
import json
import re
import os
import random
from collections import Counter

INPUT_FILE = "data/dialogue_acts/meddialog_raw.json"
OUTPUT_FILE = "data/dialogue_acts/meddialog_train.json"

def auto_label(utterance: str) -> str:
    """Heuristic auto-labeler for full doctor/patient utterances."""
    # Strip speaker prefix for classification
    if ":" in utterance:
        role, content = utterance.split(":", 1)
        role = role.strip().lower()
    else:
        role = ""
        content = utterance
    
    t = content.strip().lower()
    if not t:
        return "OTHER"

    # 1. QUESTIONS
    if t.endswith("?") or re.match(
        r"^(how|what|when|where|why|who|which|do |does |did |is |are |was |were |"
        r"should |could |would |can |will |have |has |had |could you|can you|do you|would you)", t
    ):
        return "QUESTION"

    # 2. TREATMENT PLANS
    if re.search(r"\b(take |dose |mg |tablets?|capsul|prescri|treatment|rest |fluids|quarantine|"
                 r"isolate|avoid |stop |apply |use |schedule|start |begin|drink |hydrate|cure|"
                 r'better to|treatment of choice|consult doctor|as early as possible|" )', t):
        return "TREATMENT_PLAN"

    # 3. REASSURANCE
    if re.search(r"\b(low risk|pass without|mildly ill|safe|don.?t worry|common|usually|"
                 r"nothing serious|no need to worry|typically will|don.t have to|wait for now|stay safe)", t):
        return "REASSURANCE"

    # 4. DIAGNOSIS
    if re.search(r"\b(diagnosed|signs of|indicates|consistent with|suggests|likely|probably|"
                 r"test result|culture |x[- ]ray|mri|scan|confirms|reveals|assessment|"
                 r"top symptoms include|symptoms include|is airborne|primarily an|possibility|suspect case)", t):
        return "DIAGNOSIS_STATEMENT"

    # 5. HISTORY
    if re.search(r"\b(history|past|years ago|weeks|months|since|chronic|previously|used to|"
                 r"quit|family|allerg|surger|procedure|medications|smoke|deliver|pregnant|"
                 r"pregnancy|miscarried|gave birth|fertility|pcos|weak cervix|level is low|shots every week)", t):
        return "HISTORY"

    # 6. SYMPTOM REPORT (default for patient)
    if role == "patient":
        return "SYMPTOM_REPORT"

    # 7. OTHER (admin, greetings, etc.)
    return "OTHER"

def split_into_sentences(text: str) -> list:
    """Split text into individual sentences."""
    if ":" in text:
        _, content = text.split(":", 1)
    else:
        content = text
    
    # Split on sentence boundaries but keep periods in abbreviations
    sentences = re.split(r'(?<=[.!?])\s+', content.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 8]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Place your MedDialog JSON here.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} conversations from MedDialog...")
    
    all_examples = []
    
    for convo in raw_data:
        for utterance in convo.get("utterances", []):
            label = auto_label(utterance)
            sentences = split_into_sentences(utterance)
            for sent in sentences:
                all_examples.append({"text": sent, "label": label})

    # Shuffle for training stability
    random.seed(42)
    random.shuffle(all_examples)
    
    counts = Counter(ex["label"] for ex in all_examples)
    print(f"\nParsed {len(all_examples)} training sentences.")
    print("\nLabel Distribution:")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:25s}: {count}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
