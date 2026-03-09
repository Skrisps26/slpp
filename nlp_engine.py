"""
MedScribe – Medical NLP Engine
Primary: spaCy + scispaCy for biomedical NER
Negation: sentence-boundary-aware rule-based (robust, no extra deps)
Enhancement: curated colloquial→clinical lexicon
Fallback: pure rule-based if spaCy models not installed
"""

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Symptom:
    name: str
    negated: bool = False
    severity: Optional[str] = None
    duration: Optional[str] = None
    frequency: Optional[str] = None
    location: Optional[str] = None
    character: Optional[str] = None
    context: str = ""
    source: str = "rule"   # "spacy" | "rule" | "colloquial"

@dataclass
class Medication:
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    status: str = "mentioned"

@dataclass
class Vital:
    name: str
    value: str
    unit: str
    raw: str

@dataclass
class Diagnosis:
    name: str
    icd10: Optional[str] = None
    certainty: str = "possible"
    primary: bool = False

@dataclass
class ClinicalEntities:
    symptoms: List[Symptom] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)
    vitals: List[Vital] = field(default_factory=list)
    diagnoses: List[Diagnosis] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    social_history: dict = field(default_factory=dict)
    review_of_systems: dict = field(default_factory=dict)
    assessment_notes: List[str] = field(default_factory=list)
    plan_items: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# spaCy loader
# ═══════════════════════════════════════════════════════════════════════════

def _load_spacy():
    """
    Load best available scispaCy/spaCy model.
    Preference order:
      1. en_ner_bc5cdr_md  – BC5CDR: diseases + chemicals, best for clinical NER
      2. en_core_sci_md    – general biomedical
      3. en_core_sci_sm    – smallest scispaCy
      4. en_core_web_sm    – base spaCy (generic)
      5. None              – pure rule-based fallback
    Returns (nlp, model_name) or (None, None).
    """
    try:
        import spacy
    except ImportError:
        return None, None

    for model in ["en_ner_bc5cdr_md", "en_core_sci_md", "en_core_sci_sm", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model)
            return nlp, model
        except OSError:
            continue
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Colloquial → Clinical lexicon
# ═══════════════════════════════════════════════════════════════════════════

COLLOQUIAL_TO_CLINICAL = {
    # Head / neuro
    "headache": "Cephalgia", "head hurts": "Cephalgia",
    "head is killing me": "Severe cephalgia", "migraine": "Migraine",
    "head is pounding": "Cephalgia (pulsating)", "splitting headache": "Severe cephalgia",
    "throbbing head": "Cephalgia (pulsating)", "brain fog": "Cognitive impairment / brain fog",
    "can't think straight": "Cognitive impairment",
    "dizzy": "Dizziness", "dizziness": "Dizziness", "feel dizzy": "Dizziness",
    "room spinning": "Vertigo", "world is spinning": "Vertigo", "vertigo": "Vertigo",
    "passed out": "Syncope", "fainted": "Syncope", "blacked out": "Syncope",
    "nearly fainted": "Pre-syncope", "lightheaded": "Pre-syncope / lightheadedness",
    "feel faint": "Pre-syncope", "fainting": "Syncope",
    "pins and needles": "Paresthesia", "tingling": "Paresthesia",
    "numbness": "Numbness / paresthesia", "numb": "Numbness / paresthesia",
    "weakness": "Weakness", "weak": "Weakness",
    "tremor": "Tremor", "shaking": "Tremor", "hands shaking": "Tremor",
    "seizure": "Seizure", "fit": "Seizure", "blackout": "Seizure / syncope",
    "confused": "Confusion / altered mentation", "confusion": "Confusion / altered mentation",
    "out of it": "Altered mentation", "memory loss": "Amnesia / memory impairment",
    "forgetting things": "Memory impairment", "can't remember": "Memory impairment",
    "blurred vision": "Blurred vision", "blurry vision": "Blurred vision",
    "double vision": "Diplopia", "seeing double": "Diplopia",
    "ringing in ears": "Tinnitus", "ears ringing": "Tinnitus", "tinnitus": "Tinnitus",
    "hearing loss": "Hearing loss", "can't hear": "Hearing loss",

    # Chest / cardiac
    "chest pain": "Chest pain", "heart hurts": "Chest pain / cardiac",
    "chest tightness": "Chest tightness", "tight chest": "Chest tightness",
    "pressure in chest": "Chest pressure", "chest pressure": "Chest pressure",
    "heart racing": "Tachycardia / palpitations", "racing heart": "Tachycardia / palpitations",
    "heart pounding": "Palpitations", "heart beating fast": "Tachycardia / palpitations",
    "heart skipping": "Palpitations / dysrhythmia", "heart fluttering": "Palpitations",
    "palpitations": "Palpitations", "irregular heartbeat": "Dysrhythmia",
    "shortness of breath": "Dyspnea", "short of breath": "Dyspnea",
    "can't catch my breath": "Dyspnea", "out of breath": "Dyspnea",
    "winded": "Exertional dyspnea", "breathlessness": "Dyspnea",
    "difficulty breathing": "Dyspnea", "trouble breathing": "Dyspnea",
    "can't breathe": "Dyspnea",
    "leg swelling": "Lower extremity edema", "ankle swelling": "Ankle edema",
    "puffy ankles": "Ankle edema", "puffy legs": "Lower extremity edema",
    "calf pain": "Calf pain / possible DVT",

    # Respiratory
    "cough": "Cough", "dry cough": "Dry cough",
    "wet cough": "Productive cough", "productive cough": "Productive cough",
    "coughing up mucus": "Productive cough with sputum",
    "coughing up phlegm": "Productive cough with sputum",
    "coughing up blood": "Hemoptysis", "coughing blood": "Hemoptysis",
    "hemoptysis": "Hemoptysis", "wheezing": "Wheezing",
    "runny nose": "Rhinorrhea", "nose is running": "Rhinorrhea",
    "stuffy nose": "Nasal congestion", "blocked nose": "Nasal congestion",
    "nasal congestion": "Nasal congestion", "sneezing": "Sneezing",
    "postnasal drip": "Postnasal drip", "hoarseness": "Dysphonia",
    "lost my voice": "Dysphonia / aphonia",
    "nosebleed": "Epistaxis", "nose is bleeding": "Epistaxis",
    "snoring": "Snoring / possible sleep apnea",

    # GI / abdominal
    "nausea": "Nausea", "queasy": "Nausea", "feel like throwing up": "Nausea",
    "feel sick": "Nausea / malaise", "upset stomach": "Dyspepsia / nausea",
    "vomiting": "Vomiting", "throwing up": "Vomiting", "puking": "Vomiting",
    "been sick": "Vomiting / nausea", "vomited": "Vomiting",
    "can't keep food down": "Vomiting / nausea",
    "abdominal pain": "Abdominal pain", "stomach pain": "Abdominal pain",
    "tummy ache": "Abdominal pain", "tummy hurts": "Abdominal pain",
    "stomach ache": "Abdominal pain", "stomach is killing me": "Severe abdominal pain",
    "belly pain": "Abdominal pain", "belly ache": "Abdominal pain",
    "stomach cramps": "Abdominal cramping", "gut pain": "Abdominal pain",
    "diarrhea": "Diarrhea", "loose stool": "Diarrhea", "loose stools": "Diarrhea",
    "watery stool": "Diarrhea", "the runs": "Diarrhea",
    "constipation": "Constipation", "can't poop": "Constipation",
    "blood in stool": "Hematochezia", "blood in poop": "Hematochezia",
    "blood when i poop": "Hematochezia", "rectal bleeding": "Hematochezia",
    "black stool": "Melena", "black poop": "Melena",
    "heartburn": "Pyrosis / heartburn", "acid reflux": "Gastroesophageal reflux",
    "indigestion": "Dyspepsia",
    "bloating": "Abdominal distension / bloating", "bloated": "Abdominal distension / bloating",
    "gassy": "Flatulence", "gas": "Flatulence",
    "burping": "Eructation / belching",
    "loss of appetite": "Anorexia", "no appetite": "Anorexia",
    "not hungry": "Decreased appetite",
    "jaundice": "Jaundice",
    "difficulty swallowing": "Dysphagia", "hard to swallow": "Dysphagia",
    "dysphagia": "Dysphagia", "lump in throat": "Globus sensation / dysphagia",

    # GU / genitourinary
    "balls hurt": "Orchialgia / testicular pain",
    "balls are hurting": "Orchialgia / testicular pain",
    "balls hurting": "Orchialgia / testicular pain",
    "ball pain": "Orchialgia / testicular pain",
    "testicle pain": "Orchialgia / testicular pain",
    "testicles hurt": "Orchialgia / testicular pain",
    "nuts hurt": "Orchialgia / testicular pain",
    "testicular pain": "Orchialgia / testicular pain",
    "scrotal pain": "Scrotal pain", "scrotal swelling": "Scrotal swelling",
    "swollen balls": "Scrotal / testicular swelling",
    "swollen testicle": "Testicular swelling",
    "dick hurts": "Penile pain", "penis hurts": "Penile pain",
    "penile pain": "Penile pain", "tip hurts": "Urethral / penile discomfort",
    "penile discharge": "Urethral discharge", "discharge from penis": "Urethral discharge",
    "vagina hurts": "Vaginal pain", "vaginal pain": "Vaginal pain",
    "vaginal discharge": "Vaginal discharge",
    "down there hurts": "Genital / pelvic pain", "private parts hurt": "Genital / pelvic pain",
    "itchy down there": "Genital pruritus", "burning down there": "Dysuria / urogenital burning",
    "pelvic pain": "Pelvic pain", "groin pain": "Inguinal / groin pain",
    "period pain": "Dysmenorrhea", "period cramps": "Dysmenorrhea",
    "painful periods": "Dysmenorrhea", "heavy periods": "Menorrhagia",
    "irregular periods": "Menstrual irregularity", "missed period": "Amenorrhea",
    "spotting": "Intermenstrual bleeding / spotting",
    "burning when i pee": "Dysuria", "burns when i pee": "Dysuria",
    "hurts to pee": "Dysuria", "painful urination": "Dysuria",
    "burning urination": "Dysuria", "dysuria": "Dysuria",
    "frequent urination": "Urinary frequency / pollakiuria",
    "peeing a lot": "Urinary frequency / pollakiuria",
    "pee a lot": "Urinary frequency / pollakiuria",
    "going to the bathroom a lot": "Urinary frequency / pollakiuria",
    "up at night to pee": "Nocturia", "waking up to pee": "Nocturia", "nocturia": "Nocturia",
    "can't hold my pee": "Urinary incontinence", "leaking urine": "Urinary incontinence",
    "incontinence": "Urinary incontinence",
    "blood in urine": "Hematuria", "blood in pee": "Hematuria",
    "blood when i pee": "Hematuria", "red urine": "Hematuria",
    "cloudy urine": "Pyuria / cloudy urine",
    "urinary urgency": "Urinary urgency",
    "erectile dysfunction": "Erectile dysfunction",
    "can't get an erection": "Erectile dysfunction",

    # Constitutional
    "fever": "Pyrexia / fever", "high temperature": "Pyrexia / fever",
    "running a fever": "Pyrexia / fever", "running a temperature": "Pyrexia / fever",
    "chills": "Chills / rigors", "shivering": "Chills / rigors",
    "night sweats": "Night sweats", "sweating at night": "Night sweats",
    "waking up sweating": "Night sweats", "sweating a lot": "Diaphoresis",
    "fatigue": "Fatigue", "tiredness": "Fatigue", "tired": "Fatigue",
    "exhaustion": "Fatigue / exhaustion", "exhausted": "Fatigue / exhaustion",
    "wiped out": "Fatigue / exhaustion", "drained": "Fatigue",
    "no energy": "Fatigue / lethargy", "always tired": "Chronic fatigue",
    "malaise": "Malaise", "feel terrible": "Malaise / general unwellness",
    "feel awful": "Malaise", "feel unwell": "Malaise / general unwellness",
    "under the weather": "Malaise / general unwellness",
    "not feeling well": "Malaise / general unwellness",
    "weight loss": "Unintentional weight loss", "losing weight": "Unintentional weight loss",
    "weight gain": "Weight gain",
    "swelling": "Edema / swelling", "swollen": "Edema / swelling",
    "edema": "Edema", "puffy": "Edema",

    # MSK / skin
    "back pain": "Dorsalgia", "back is killing me": "Severe dorsalgia",
    "neck pain": "Cervicalgia", "shoulder pain": "Shoulder pain",
    "knee pain": "Knee pain", "knee is killing me": "Severe knee pain",
    "hip pain": "Hip pain", "wrist pain": "Wrist pain", "ankle pain": "Ankle pain",
    "joint pain": "Arthralgia", "muscle pain": "Myalgia", "muscle aches": "Myalgia",
    "stiffness": "Joint stiffness", "morning stiffness": "Morning stiffness",
    "swollen joints": "Joint swelling / synovitis",
    "rash": "Skin rash / dermatitis", "hives": "Urticaria",
    "itching": "Pruritus", "itchy": "Pruritus",
    "bruising": "Ecchymosis / bruising",
    "hair loss": "Alopecia", "losing hair": "Alopecia",
    "lump": "Mass / lump — requires evaluation",
    "swollen glands": "Lymphadenopathy", "swollen lymph nodes": "Lymphadenopathy",
    "sore throat": "Pharyngitis / sore throat", "throat hurts": "Pharyngitis / sore throat",
    "ear pain": "Otalgia", "earache": "Otalgia",
    "eye pain": "Ophthalmalgia", "red eyes": "Conjunctival injection / red eye",
    "pink eye": "Conjunctivitis",

    # Psych
    "anxiety": "Anxiety", "anxious": "Anxiety",
    "on edge": "Anxiety", "worrying a lot": "Anxiety / excessive worry",
    "panic attacks": "Panic attacks", "panic attack": "Panic attacks",
    "depression": "Depressive symptoms", "depressed": "Depressive symptoms",
    "feeling low": "Depressive symptoms", "feeling down": "Depressive symptoms",
    "feel hopeless": "Depressive symptoms", "no motivation": "Depressive symptoms / amotivation",
    "can't enjoy things": "Anhedonia", "lost interest": "Anhedonia",
    "insomnia": "Insomnia", "can't sleep": "Insomnia", "trouble sleeping": "Insomnia",
    "sleep problems": "Sleep disturbance",
    "mood swings": "Mood disturbance / emotional lability",
    "irritability": "Irritability", "irritable": "Irritability",
    "stress": "Psychosocial stress", "stressed": "Psychosocial stress",
    "burnt out": "Burnout / chronic stress",
    "suicidal thoughts": "Suicidal ideation",
    "self harm": "Self-harm",
    "hearing voices": "Auditory hallucinations",
    "seeing things": "Visual hallucinations",

    # Endocrine
    "excessive thirst": "Polydipsia", "always thirsty": "Polydipsia",
    "really thirsty": "Polydipsia", "increased thirst": "Polydipsia",
    "excessive urination": "Polyuria",
    "hot flashes": "Hot flashes / vasomotor symptoms",
    "hot flushes": "Hot flashes / vasomotor symptoms",
    "cold intolerance": "Cold intolerance", "always cold": "Cold intolerance",
    "heat intolerance": "Heat intolerance",
    "feeling shaky": "Tremor / possible hypoglycemia",
    "shaky": "Tremor / possible hypoglycemia",
}

# Dynamic regex patterns for "my X hurts/aches/is killing me" sentences
_PAIN_V = r"(?:hurt(?:s|ing)?|ache?s?|aching|is killing me|kill(?:s|ing) me|is sore|is tender|is throbbing|is burning|burning|is painful|in pain|are painful|are sore)"
_SWELL_V = r"(?:is swollen|are swollen|has swollen|swelled up|swelling)"

COLLOQUIAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:my\s+)?(?:balls?|testicles?|nuts?|scrotum)\s*" + _PAIN_V, re.I),
     "Orchialgia / testicular pain"),
    (re.compile(r"\b(?:my\s+)?(?:balls?|testicles?|nuts?)\s*" + _SWELL_V, re.I),
     "Testicular / scrotal swelling"),
    (re.compile(r"\b(?:my\s+)?(?:dick|penis|willy)\s*" + _PAIN_V, re.I),
     "Penile pain"),
    (re.compile(r"\b(?:discharge|dripping|leaking)\s+from\s+(?:my\s+)?(?:dick|penis|willy)", re.I),
     "Urethral discharge"),
    (re.compile(r"\b(?:my\s+)?(?:vagina|vulva|vag|privates?)\s*" + _PAIN_V, re.I),
     "Vaginal / vulvar pain"),
    (re.compile(r"\bdown\s+(?:there|below)\s*" + _PAIN_V, re.I),
     "Genital / pelvic pain"),
    (re.compile(r"\b(?:my\s+)?(?:stomach|tummy|belly|gut|abdomen)\s*" + _PAIN_V, re.I),
     "Abdominal pain"),
    (re.compile(r"\b(?:my\s+)?(?:stomach|tummy|belly)\s+(?:is\s+)?(?:bloated|swollen|distended)", re.I),
     "Abdominal distension / bloating"),
    (re.compile(r"\b(?:my\s+)?(?:back|spine|lower back|lumbar)\s*" + _PAIN_V, re.I),
     "Dorsalgia"),
    (re.compile(r"\b(?:my\s+)?(?:chest|sternum|ribs?)\s*" + _PAIN_V, re.I),
     "Chest pain"),
    (re.compile(r"\b(?:my\s+)?(?:head|skull)\s*" + _PAIN_V, re.I),
     "Cephalgia"),
    (re.compile(r"\b(?:my\s+)?(?:neck)\s*" + _PAIN_V, re.I),
     "Cervicalgia / neck pain"),
    (re.compile(r"\b(?:my\s+)?(?:knee|knees)\s*" + _PAIN_V, re.I),
     "Knee pain"),
    (re.compile(r"\b(?:my\s+)?(?:hip|hips)\s*" + _PAIN_V, re.I),
     "Hip pain"),
    (re.compile(r"\b(?:my\s+)?(?:shoulder|shoulders)\s*" + _PAIN_V, re.I),
     "Shoulder pain"),
    (re.compile(r"\b(?:my\s+)?(?:ankle|ankles)\s*" + _PAIN_V, re.I),
     "Ankle pain"),
    (re.compile(r"\b(?:my\s+)?(?:elbow|elbows)\s*" + _PAIN_V, re.I),
     "Elbow pain"),
    (re.compile(r"\b(?:my\s+)?(?:foot|feet|toe|toes)\s*" + _PAIN_V, re.I),
     "Foot / toe pain"),
    (re.compile(r"\b(?:my\s+)?(?:finger|fingers|thumb)\s*" + _PAIN_V, re.I),
     "Hand / finger pain"),
    (re.compile(r"\b(?:my\s+)?(?:wrist|wrists)\s*" + _PAIN_V, re.I),
     "Wrist pain"),
    (re.compile(r"\b(?:my\s+)?(?:ear|ears)\s*" + _PAIN_V, re.I),
     "Otalgia"),
    (re.compile(r"\b(?:my\s+)?(?:eye|eyes)\s*" + _PAIN_V, re.I),
     "Ocular pain"),
    (re.compile(r"\b(?:it\s+)?(?:hurts?|burns?|stings?)\s+when\s+(?:i\s+)?(?:pee|urinate)", re.I),
     "Dysuria"),
    (re.compile(r"\b(?:burning|pain)\s+when\s+(?:peeing|urinating)", re.I),
     "Dysuria"),
    (re.compile(r"\b(?:it\s+)?(?:hurts?|bleeds?)\s+when\s+(?:i\s+)?(?:poop|defecate)", re.I),
     "Painful defecation / rectal pain"),
    (re.compile(r"\bblood\s+(?:when|after)\s+(?:i\s+)?(?:wipe|poop)", re.I),
     "Hematochezia / rectal bleeding"),
    (re.compile(r"\b(?:can't|cannot|hard to|trouble)\s+(?:breathe|breathing|catch\s+(?:my\s+)?breath)", re.I),
     "Dyspnea"),
    (re.compile(r"\b(?:can't|cannot)\s+(?:sleep|fall\s+asleep|stay\s+asleep)", re.I),
     "Insomnia"),
    (re.compile(r"\b(?:found|feel|noticed)\s+(?:a\s+)?(?:lump|bump|mass|growth)\s+(?:in|on|near|under)\s+(?:my\s+)?\w+", re.I),
     "Mass / lump — requires evaluation"),
    (re.compile(r"\b(?:my\s+)?(?:glands?|lymph\s+nodes?)\s+(?:are\s+)?(?:swollen|enlarged)", re.I),
     "Lymphadenopathy"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Medication lexicon
# ═══════════════════════════════════════════════════════════════════════════

MEDICATION_TERMS = {
    "ibuprofen": "NSAID analgesic", "naproxen": "NSAID analgesic",
    "aspirin": "NSAID / antiplatelet", "acetaminophen": "Analgesic / antipyretic",
    "tylenol": "Analgesic / antipyretic", "advil": "NSAID (ibuprofen)",
    "motrin": "NSAID (ibuprofen)", "aleve": "NSAID (naproxen)",
    "tramadol": "Opioid analgesic", "oxycodone": "Opioid analgesic",
    "hydrocodone": "Opioid analgesic", "morphine": "Opioid analgesic",
    "codeine": "Opioid analgesic", "gabapentin": "Anticonvulsant / neuropathic pain",
    "pregabalin": "Anticonvulsant / neuropathic pain",
    "amoxicillin": "Penicillin antibiotic", "augmentin": "Penicillin / beta-lactamase inhibitor",
    "azithromycin": "Macrolide antibiotic", "z-pack": "Macrolide antibiotic",
    "zpack": "Macrolide antibiotic", "doxycycline": "Tetracycline antibiotic",
    "ciprofloxacin": "Fluoroquinolone antibiotic", "cipro": "Fluoroquinolone antibiotic",
    "levofloxacin": "Fluoroquinolone antibiotic", "metronidazole": "Antibiotic / antiprotozoal",
    "flagyl": "Antibiotic (metronidazole)", "cephalexin": "Cephalosporin antibiotic",
    "clindamycin": "Lincosamide antibiotic", "bactrim": "Sulfonamide antibiotic",
    "nitrofurantoin": "Antibiotic (UTI)", "macrobid": "Antibiotic (nitrofurantoin)",
    "penicillin": "Penicillin antibiotic",
    "lisinopril": "ACE inhibitor", "enalapril": "ACE inhibitor",
    "amlodipine": "Calcium channel blocker",
