"""
nlp_engine/ontology.py
Anatomical ontology and body region reasoning.
Prevents incorrect body-region assignments (e.g. orchialgia → chest).
"""

from typing import Optional

# Hierarchical anatomical ontology
ANATOMICAL_HIERARCHY = {
    "head": {
        "children": ["scalp", "skull", "face", "temporomandibular joint"],
        "symptoms": ["headache", "cephalgia", "facial pain", "jaw pain", "scalp tenderness"],
    },
    "neck": {
        "children": ["cervical spine", "throat", "thyroid"],
        "symptoms": ["neck pain", "cervicalgia", "sore throat", "pharyngitis", "dysphagia", "hoarseness"],
    },
    "thorax": {
        "children": ["heart", "lungs", "esophagus", "chest wall", "pleura", "mediastinum"],
        "symptoms": ["chest pain", "chest pressure", "chest tightness", "pleuritic pain", "palpitations", "shortness of breath"],
    },
    "abdomen": {
        "children": ["stomach", "liver", "gallbladder", "spleen", "small intestine", "colon", "pancreas", "appendix"],
        "symptoms": ["abdominal pain", "epigastric pain", "right upper quadrant pain", "left upper quadrant pain", "nausea", "heartburn", "bloating"],
    },
    "pelvis": {
        "children": ["bladder", "uterus", "ovaries", "prostate", "rectum", "sigmoid colon"],
        "symptoms": ["pelvic pain", "lower abdominal pain", "dysuria", "urinary frequency", "menstrual pain"],
    },
    "genitalia": {
        "children": ["penis", "testes", "scrotum", "vulva", "vagina"],
        "symptoms": ["testicular pain", "orchialgia", "scrotal pain", "penile pain", "vaginal pain", "genital pain"],
    },
    "back": {
        "children": ["lumbar spine", "thoracic spine", "sacrum", "coccyx"],
        "symptoms": ["back pain", "dorsalgia", "lumbar pain", "sciatica", "radiculopathy"],
    },
    "upper_extremity": {
        "children": ["shoulder", "arm", "elbow", "forearm", "wrist", "hand", "fingers"],
        "symptoms": ["shoulder pain", "arm pain", "elbow pain", "wrist pain", "hand pain", "finger pain"],
    },
    "lower_extremity": {
        "children": ["hip", "thigh", "knee", "leg", "calf", "ankle", "foot", "toes"],
        "symptoms": ["hip pain", "knee pain", "calf pain", "ankle pain", "foot pain", "leg swelling"],
    },
    "systemic": {
        "children": [],
        "symptoms": ["fever", "fatigue", "weight loss", "weight gain", "chills", "night sweats", "diaphoresis", "malaise"],
    },
}

# Explicit symptom → body region mapping (prevents misclassification)
SYMPTOM_TO_REGION = {
    # GU
    "orchialgia": "genitalia",
    "testicular pain": "genitalia",
    "scrotal pain": "genitalia",
    "penile pain": "genitalia",
    "vaginal pain": "pelvis",
    "pelvic pain": "pelvis",
    "dysuria": "pelvis",
    "urinary frequency": "pelvis",
    # Thorax
    "chest pain": "thorax",
    "chest pressure": "thorax",
    "chest tightness": "thorax",
    "palpitations": "thorax",
    # Abdomen
    "abdominal pain": "abdomen",
    "nausea": "abdomen",
    "heartburn": "abdomen",
    "diarrhea": "abdomen",
    "constipation": "abdomen",
    # Neuro/Head
    "headache": "head",
    "cephalgia": "head",
    "dizziness": "head",
    "vertigo": "head",
    # MSK
    "back pain": "back",
    "neck pain": "neck",
    "knee pain": "lower_extremity",
    "hip pain": "lower_extremity",
    "calf pain": "lower_extremity",
    "ankle pain": "lower_extremity",
    "shoulder pain": "upper_extremity",
    "wrist pain": "upper_extremity",
    # Systemic
    "fever": "systemic",
    "fatigue": "systemic",
    "weight loss": "systemic",
    "night sweats": "systemic",
}


def get_body_region(symptom_name: str) -> Optional[str]:
    """
    Return the canonical body region for a symptom.
    Prevents misclassification (e.g. orchialgia should not map to 'chest').
    """
    lower = symptom_name.lower()

    # Direct lookup
    for key, region in SYMPTOM_TO_REGION.items():
        if key in lower:
            return region

    # Search hierarchy
    for region, data in ANATOMICAL_HIERARCHY.items():
        for sym in data.get("symptoms", []):
            if sym in lower or lower in sym:
                return region

    return "systemic"


def get_related_regions(region: str) -> list:
    """Return the sub-regions of a body region."""
    return ANATOMICAL_HIERARCHY.get(region, {}).get("children", [])


def region_to_icd_prefix(region: str) -> Optional[str]:
    """Return the ICD-10 chapter prefix for a body region."""
    mapping = {
        "thorax": "I",        # Cardiovascular
        "respiratory": "J",   # Respiratory
        "abdomen": "K",       # Gastrointestinal
        "pelvis": "N",        # Genitourinary
        "genitalia": "N",
        "back": "M",          # Musculoskeletal
        "upper_extremity": "M",
        "lower_extremity": "M",
        "head": "G",          # Neurological / R (symptoms)
        "systemic": "R",
    }
    return mapping.get(region)
