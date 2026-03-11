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
from typing import List, Optional, Tuple

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
    source: str = "rule"  # "spacy" | "rule" | "colloquial"


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

    for model in [
        "en_ner_bc5cdr_md",
        "en_core_sci_md",
        "en_core_sci_sm",
        "en_core_web_sm",
    ]:
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
    "headache": "Cephalgia",
    "head hurts": "Cephalgia",
    "head is killing me": "Severe cephalgia",
    "migraine": "Migraine",
    "head is pounding": "Cephalgia (pulsating)",
    "splitting headache": "Severe cephalgia",
    "throbbing head": "Cephalgia (pulsating)",
    "brain fog": "Cognitive impairment / brain fog",
    "can't think straight": "Cognitive impairment",
    "dizzy": "Dizziness",
    "dizziness": "Dizziness",
    "feel dizzy": "Dizziness",
    "room spinning": "Vertigo",
    "world is spinning": "Vertigo",
    "vertigo": "Vertigo",
    "passed out": "Syncope",
    "fainted": "Syncope",
    "blacked out": "Syncope",
    "nearly fainted": "Pre-syncope",
    "lightheaded": "Pre-syncope / lightheadedness",
    "feel faint": "Pre-syncope",
    "fainting": "Syncope",
    "pins and needles": "Paresthesia",
    "tingling": "Paresthesia",
    "numbness": "Numbness / paresthesia",
    "numb": "Numbness / paresthesia",
    "weakness": "Weakness",
    "weak": "Weakness",
    "tremor": "Tremor",
    "shaking": "Tremor",
    "hands shaking": "Tremor",
    "seizure": "Seizure",
    "fit": "Seizure",
    "blackout": "Seizure / syncope",
    "confused": "Confusion / altered mentation",
    "confusion": "Confusion / altered mentation",
    "out of it": "Altered mentation",
    "memory loss": "Amnesia / memory impairment",
    "forgetting things": "Memory impairment",
    "can't remember": "Memory impairment",
    "blurred vision": "Blurred vision",
    "blurry vision": "Blurred vision",
    "double vision": "Diplopia",
    "seeing double": "Diplopia",
    "ringing in ears": "Tinnitus",
    "ears ringing": "Tinnitus",
    "tinnitus": "Tinnitus",
    "hearing loss": "Hearing loss",
    "can't hear": "Hearing loss",
    # Chest / cardiac
    "chest pain": "Chest pain",
    "heart hurts": "Chest pain / cardiac",
    "chest tightness": "Chest tightness",
    "tight chest": "Chest tightness",
    "pressure in chest": "Chest pressure",
    "chest pressure": "Chest pressure",
    "heart racing": "Tachycardia / palpitations",
    "racing heart": "Tachycardia / palpitations",
    "heart pounding": "Palpitations",
    "heart beating fast": "Tachycardia / palpitations",
    "heart skipping": "Palpitations / dysrhythmia",
    "heart fluttering": "Palpitations",
    "palpitations": "Palpitations",
    "irregular heartbeat": "Dysrhythmia",
    "shortness of breath": "Dyspnea",
    "short of breath": "Dyspnea",
    "can't catch my breath": "Dyspnea",
    "out of breath": "Dyspnea",
    "winded": "Exertional dyspnea",
    "breathlessness": "Dyspnea",
    "difficulty breathing": "Dyspnea",
    "trouble breathing": "Dyspnea",
    "can't breathe": "Dyspnea",
    "leg swelling": "Lower extremity edema",
    "ankle swelling": "Ankle edema",
    "puffy ankles": "Ankle edema",
    "puffy legs": "Lower extremity edema",
    "calf pain": "Calf pain / possible DVT",
    # Respiratory
    "cough": "Cough",
    "dry cough": "Dry cough",
    "wet cough": "Productive cough",
    "productive cough": "Productive cough",
    "coughing up mucus": "Productive cough with sputum",
    "coughing up phlegm": "Productive cough with sputum",
    "coughing up blood": "Hemoptysis",
    "coughing blood": "Hemoptysis",
    "hemoptysis": "Hemoptysis",
    "wheezing": "Wheezing",
    "runny nose": "Rhinorrhea",
    "nose is running": "Rhinorrhea",
    "stuffy nose": "Nasal congestion",
    "blocked nose": "Nasal congestion",
    "nasal congestion": "Nasal congestion",
    "sneezing": "Sneezing",
    "postnasal drip": "Postnasal drip",
    "hoarseness": "Dysphonia",
    "lost my voice": "Dysphonia / aphonia",
    "nosebleed": "Epistaxis",
    "nose is bleeding": "Epistaxis",
    "snoring": "Snoring / possible sleep apnea",
    # GI / abdominal
    "nausea": "Nausea",
    "queasy": "Nausea",
    "feel like throwing up": "Nausea",
    "feel sick": "Nausea / malaise",
    "upset stomach": "Dyspepsia / nausea",
    "vomiting": "Vomiting",
    "throwing up": "Vomiting",
    "puking": "Vomiting",
    "been sick": "Vomiting / nausea",
    "vomited": "Vomiting",
    "can't keep food down": "Vomiting / nausea",
    "abdominal pain": "Abdominal pain",
    "stomach pain": "Abdominal pain",
    "tummy ache": "Abdominal pain",
    "tummy hurts": "Abdominal pain",
    "stomach ache": "Abdominal pain",
    "stomach is killing me": "Severe abdominal pain",
    "belly pain": "Abdominal pain",
    "belly ache": "Abdominal pain",
    "stomach cramps": "Abdominal cramping",
    "gut pain": "Abdominal pain",
    "diarrhea": "Diarrhea",
    "loose stool": "Diarrhea",
    "loose stools": "Diarrhea",
    "watery stool": "Diarrhea",
    "the runs": "Diarrhea",
    "constipation": "Constipation",
    "can't poop": "Constipation",
    "blood in stool": "Hematochezia",
    "blood in poop": "Hematochezia",
    "blood when i poop": "Hematochezia",
    "rectal bleeding": "Hematochezia",
    "black stool": "Melena",
    "black poop": "Melena",
    "heartburn": "Pyrosis / heartburn",
    "acid reflux": "Gastroesophageal reflux",
    "indigestion": "Dyspepsia",
    "bloating": "Abdominal distension / bloating",
    "bloated": "Abdominal distension / bloating",
    "gassy": "Flatulence",
    "gas": "Flatulence",
    "burping": "Eructation / belching",
    "loss of appetite": "Anorexia",
    "no appetite": "Anorexia",
    "not hungry": "Decreased appetite",
    "jaundice": "Jaundice",
    "difficulty swallowing": "Dysphagia",
    "hard to swallow": "Dysphagia",
    "dysphagia": "Dysphagia",
    "lump in throat": "Globus sensation / dysphagia",
    # GU / genitourinary
    "balls hurt": "Orchialgia / testicular pain",
    "balls are hurting": "Orchialgia / testicular pain",
    "balls hurting": "Orchialgia / testicular pain",
    "ball pain": "Orchialgia / testicular pain",
    "testicle pain": "Orchialgia / testicular pain",
    "testicles hurt": "Orchialgia / testicular pain",
    "nuts hurt": "Orchialgia / testicular pain",
    "testicular pain": "Orchialgia / testicular pain",
    "scrotal pain": "Scrotal pain",
    "scrotal swelling": "Scrotal swelling",
    "swollen balls": "Scrotal / testicular swelling",
    "swollen testicle": "Testicular swelling",
    "dick hurts": "Penile pain",
    "penis hurts": "Penile pain",
    "penile pain": "Penile pain",
    "tip hurts": "Urethral / penile discomfort",
    "penile discharge": "Urethral discharge",
    "discharge from penis": "Urethral discharge",
    "vagina hurts": "Vaginal pain",
    "vaginal pain": "Vaginal pain",
    "vaginal discharge": "Vaginal discharge",
    "down there hurts": "Genital / pelvic pain",
    "private parts hurt": "Genital / pelvic pain",
    "itchy down there": "Genital pruritus",
    "burning down there": "Dysuria / urogenital burning",
    "pelvic pain": "Pelvic pain",
    "groin pain": "Inguinal / groin pain",
    "period pain": "Dysmenorrhea",
    "period cramps": "Dysmenorrhea",
    "painful periods": "Dysmenorrhea",
    "heavy periods": "Menorrhagia",
    "irregular periods": "Menstrual irregularity",
    "missed period": "Amenorrhea",
    "spotting": "Intermenstrual bleeding / spotting",
    "burning when i pee": "Dysuria",
    "burns when i pee": "Dysuria",
    "hurts to pee": "Dysuria",
    "painful urination": "Dysuria",
    "burning urination": "Dysuria",
    "dysuria": "Dysuria",
    "frequent urination": "Urinary frequency / pollakiuria",
    "peeing a lot": "Urinary frequency / pollakiuria",
    "pee a lot": "Urinary frequency / pollakiuria",
    "going to the bathroom a lot": "Urinary frequency / pollakiuria",
    "up at night to pee": "Nocturia",
    "waking up to pee": "Nocturia",
    "nocturia": "Nocturia",
    "can't hold my pee": "Urinary incontinence",
    "leaking urine": "Urinary incontinence",
    "incontinence": "Urinary incontinence",
    "blood in urine": "Hematuria",
    "blood in pee": "Hematuria",
    "blood when i pee": "Hematuria",
    "red urine": "Hematuria",
    "cloudy urine": "Pyuria / cloudy urine",
    "urinary urgency": "Urinary urgency",
    "erectile dysfunction": "Erectile dysfunction",
    "can't get an erection": "Erectile dysfunction",
    # Constitutional
    "fever": "Pyrexia / fever",
    "high temperature": "Pyrexia / fever",
    "running a fever": "Pyrexia / fever",
    "running a temperature": "Pyrexia / fever",
    "chills": "Chills / rigors",
    "shivering": "Chills / rigors",
    "night sweats": "Night sweats",
    "sweating at night": "Night sweats",
    "waking up sweating": "Night sweats",
    "sweating a lot": "Diaphoresis",
    "fatigue": "Fatigue",
    "tiredness": "Fatigue",
    "tired": "Fatigue",
    "exhaustion": "Fatigue / exhaustion",
    "exhausted": "Fatigue / exhaustion",
    "wiped out": "Fatigue / exhaustion",
    "drained": "Fatigue",
    "no energy": "Fatigue / lethargy",
    "always tired": "Chronic fatigue",
    "malaise": "Malaise",
    "feel terrible": "Malaise / general unwellness",
    "feel awful": "Malaise",
    "feel unwell": "Malaise / general unwellness",
    "under the weather": "Malaise / general unwellness",
    "not feeling well": "Malaise / general unwellness",
    "weight loss": "Unintentional weight loss",
    "losing weight": "Unintentional weight loss",
    "weight gain": "Weight gain",
    "swelling": "Edema / swelling",
    "swollen": "Edema / swelling",
    "edema": "Edema",
    "puffy": "Edema",
    # MSK / skin
    "back pain": "Dorsalgia",
    "back is killing me": "Severe dorsalgia",
    "neck pain": "Cervicalgia",
    "shoulder pain": "Shoulder pain",
    "knee pain": "Knee pain",
    "knee is killing me": "Severe knee pain",
    "hip pain": "Hip pain",
    "wrist pain": "Wrist pain",
    "ankle pain": "Ankle pain",
    "joint pain": "Arthralgia",
    "muscle pain": "Myalgia",
    "muscle aches": "Myalgia",
    "stiffness": "Joint stiffness",
    "morning stiffness": "Morning stiffness",
    "swollen joints": "Joint swelling / synovitis",
    "rash": "Skin rash / dermatitis",
    "hives": "Urticaria",
    "itching": "Pruritus",
    "itchy": "Pruritus",
    "bruising": "Ecchymosis / bruising",
    "hair loss": "Alopecia",
    "losing hair": "Alopecia",
    "lump": "Mass / lump — requires evaluation",
    "swollen glands": "Lymphadenopathy",
    "swollen lymph nodes": "Lymphadenopathy",
    "sore throat": "Pharyngitis / sore throat",
    "throat hurts": "Pharyngitis / sore throat",
    "ear pain": "Otalgia",
    "earache": "Otalgia",
    "eye pain": "Ophthalmalgia",
    "red eyes": "Conjunctival injection / red eye",
    "pink eye": "Conjunctivitis",
    # Psych
    "anxiety": "Anxiety",
    "anxious": "Anxiety",
    "on edge": "Anxiety",
    "worrying a lot": "Anxiety / excessive worry",
    "panic attacks": "Panic attacks",
    "panic attack": "Panic attacks",
    "depression": "Depressive symptoms",
    "depressed": "Depressive symptoms",
    "feeling low": "Depressive symptoms",
    "feeling down": "Depressive symptoms",
    "feel hopeless": "Depressive symptoms",
    "no motivation": "Depressive symptoms / amotivation",
    "can't enjoy things": "Anhedonia",
    "lost interest": "Anhedonia",
    "insomnia": "Insomnia",
    "can't sleep": "Insomnia",
    "trouble sleeping": "Insomnia",
    "sleep problems": "Sleep disturbance",
    "mood swings": "Mood disturbance / emotional lability",
    "irritability": "Irritability",
    "irritable": "Irritability",
    "stress": "Psychosocial stress",
    "stressed": "Psychosocial stress",
    "burnt out": "Burnout / chronic stress",
    "suicidal thoughts": "Suicidal ideation",
    "self harm": "Self-harm",
    "hearing voices": "Auditory hallucinations",
    "seeing things": "Visual hallucinations",
    # Endocrine
    "excessive thirst": "Polydipsia",
    "always thirsty": "Polydipsia",
    "really thirsty": "Polydipsia",
    "increased thirst": "Polydipsia",
    "excessive urination": "Polyuria",
    "hot flashes": "Hot flashes / vasomotor symptoms",
    "hot flushes": "Hot flashes / vasomotor symptoms",
    "cold intolerance": "Cold intolerance",
    "always cold": "Cold intolerance",
    "heat intolerance": "Heat intolerance",
    "feeling shaky": "Tremor / possible hypoglycemia",
    "shaky": "Tremor / possible hypoglycemia",
}

# Dynamic regex patterns for "my X hurts/aches/is killing me" sentences
_PAIN_V = r"(?:hurt(?:s|ing)?|ache?s?|aching|is killing me|kill(?:s|ing) me|is sore|is tender|is throbbing|is burning|burning|is painful|in pain|are painful|are sore)"
_SWELL_V = r"(?:is swollen|are swollen|has swollen|swelled up|swelling)"

COLLOQUIAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(?:my\s+)?(?:balls?|testicles?|nuts?|scrotum)\s*" + _PAIN_V, re.I
        ),
        "Orchialgia / testicular pain",
    ),
    (
        re.compile(r"\b(?:my\s+)?(?:balls?|testicles?|nuts?)\s*" + _SWELL_V, re.I),
        "Testicular / scrotal swelling",
    ),
    (re.compile(r"\b(?:my\s+)?(?:dick|penis|willy)\s*" + _PAIN_V, re.I), "Penile pain"),
    (
        re.compile(
            r"\b(?:discharge|dripping|leaking)\s+from\s+(?:my\s+)?(?:dick|penis|willy)",
            re.I,
        ),
        "Urethral discharge",
    ),
    (
        re.compile(r"\b(?:my\s+)?(?:vagina|vulva|vag|privates?)\s*" + _PAIN_V, re.I),
        "Vaginal / vulvar pain",
    ),
    (
        re.compile(r"\bdown\s+(?:there|below)\s*" + _PAIN_V, re.I),
        "Genital / pelvic pain",
    ),
    (
        re.compile(
            r"\b(?:my\s+)?(?:stomach|tummy|belly|gut|abdomen)\s*" + _PAIN_V, re.I
        ),
        "Abdominal pain",
    ),
    (
        re.compile(
            r"\b(?:my\s+)?(?:stomach|tummy|belly)\s+(?:is\s+)?(?:bloated|swollen|distended)",
            re.I,
        ),
        "Abdominal distension / bloating",
    ),
    (
        re.compile(r"\b(?:my\s+)?(?:back|spine|lower back|lumbar)\s*" + _PAIN_V, re.I),
        "Dorsalgia",
    ),
    (
        re.compile(r"\b(?:my\s+)?(?:chest|sternum|ribs?)\s*" + _PAIN_V, re.I),
        "Chest pain",
    ),
    (re.compile(r"\b(?:my\s+)?(?:head|skull)\s*" + _PAIN_V, re.I), "Cephalgia"),
    (re.compile(r"\b(?:my\s+)?(?:neck)\s*" + _PAIN_V, re.I), "Cervicalgia / neck pain"),
    (re.compile(r"\b(?:my\s+)?(?:knee|knees)\s*" + _PAIN_V, re.I), "Knee pain"),
    (re.compile(r"\b(?:my\s+)?(?:hip|hips)\s*" + _PAIN_V, re.I), "Hip pain"),
    (
        re.compile(r"\b(?:my\s+)?(?:shoulder|shoulders)\s*" + _PAIN_V, re.I),
        "Shoulder pain",
    ),
    (re.compile(r"\b(?:my\s+)?(?:ankle|ankles)\s*" + _PAIN_V, re.I), "Ankle pain"),
    (re.compile(r"\b(?:my\s+)?(?:elbow|elbows)\s*" + _PAIN_V, re.I), "Elbow pain"),
    (
        re.compile(r"\b(?:my\s+)?(?:foot|feet|toe|toes)\s*" + _PAIN_V, re.I),
        "Foot / toe pain",
    ),
    (
        re.compile(r"\b(?:my\s+)?(?:finger|fingers|thumb)\s*" + _PAIN_V, re.I),
        "Hand / finger pain",
    ),
    (re.compile(r"\b(?:my\s+)?(?:wrist|wrists)\s*" + _PAIN_V, re.I), "Wrist pain"),
    (re.compile(r"\b(?:my\s+)?(?:ear|ears)\s*" + _PAIN_V, re.I), "Otalgia"),
    (re.compile(r"\b(?:my\s+)?(?:eye|eyes)\s*" + _PAIN_V, re.I), "Ocular pain"),
    (
        re.compile(
            r"\b(?:it\s+)?(?:hurts?|burns?|stings?)\s+when\s+(?:i\s+)?(?:pee|urinate)",
            re.I,
        ),
        "Dysuria",
    ),
    (re.compile(r"\b(?:burning|pain)\s+when\s+(?:peeing|urinating)", re.I), "Dysuria"),
    (
        re.compile(
            r"\b(?:it\s+)?(?:hurts?|bleeds?)\s+when\s+(?:i\s+)?(?:poop|defecate)", re.I
        ),
        "Painful defecation / rectal pain",
    ),
    (
        re.compile(r"\bblood\s+(?:when|after)\s+(?:i\s+)?(?:wipe|poop)", re.I),
        "Hematochezia / rectal bleeding",
    ),
    (
        re.compile(
            r"\b(?:can't|cannot|hard to|trouble)\s+(?:breathe|breathing|catch\s+(?:my\s+)?breath)",
            re.I,
        ),
        "Dyspnea",
    ),
    (
        re.compile(r"\b(?:can't|cannot)\s+(?:sleep|fall\s+asleep|stay\s+asleep)", re.I),
        "Insomnia",
    ),
    (
        re.compile(
            r"\b(?:found|feel|noticed)\s+(?:a\s+)?(?:lump|bump|mass|growth)\s+(?:in|on|near|under)\s+(?:my\s+)?\w+",
            re.I,
        ),
        "Mass / lump — requires evaluation",
    ),
    (
        re.compile(
            r"\b(?:my\s+)?(?:glands?|lymph\s+nodes?)\s+(?:are\s+)?(?:swollen|enlarged)",
            re.I,
        ),
        "Lymphadenopathy",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Medication lexicon
# ═══════════════════════════════════════════════════════════════════════════

MEDICATION_TERMS = {
    "ibuprofen": "NSAID analgesic",
    "naproxen": "NSAID analgesic",
    "aspirin": "NSAID / antiplatelet",
    "acetaminophen": "Analgesic / antipyretic",
    "tylenol": "Analgesic / antipyretic",
    "advil": "NSAID (ibuprofen)",
    "motrin": "NSAID (ibuprofen)",
    "aleve": "NSAID (naproxen)",
    "tramadol": "Opioid analgesic",
    "oxycodone": "Opioid analgesic",
    "hydrocodone": "Opioid analgesic",
    "morphine": "Opioid analgesic",
    "codeine": "Opioid analgesic",
    "gabapentin": "Anticonvulsant / neuropathic pain",
    "pregabalin": "Anticonvulsant / neuropathic pain",
    "amoxicillin": "Penicillin antibiotic",
    "augmentin": "Penicillin / beta-lactamase inhibitor",
    "azithromycin": "Macrolide antibiotic",
    "z-pack": "Macrolide antibiotic",
    "zpack": "Macrolide antibiotic",
    "doxycycline": "Tetracycline antibiotic",
    "ciprofloxacin": "Fluoroquinolone antibiotic",
    "cipro": "Fluoroquinolone antibiotic",
    "levofloxacin": "Fluoroquinolone antibiotic",
    "metronidazole": "Antibiotic / antiprotozoal",
    "flagyl": "Antibiotic (metronidazole)",
    "cephalexin": "Cephalosporin antibiotic",
    "clindamycin": "Lincosamide antibiotic",
    "bactrim": "Sulfonamide antibiotic",
    "nitrofurantoin": "Antibiotic (UTI)",
    "macrobid": "Antibiotic (nitrofurantoin)",
    "penicillin": "Penicillin antibiotic",
    "lisinopril": "ACE inhibitor",
    "enalapril": "ACE inhibitor",
    "amlodipine": "Calcium channel blocker",
    "norvasc": "Calcium channel blocker",
    "metoprolol": "Beta-blocker",
    "atenolol": "Beta-blocker",
    "carvedilol": "Beta-blocker",
    "losartan": "ARB antihypertensive",
    "valsartan": "ARB antihypertensive",
    "hydrochlorothiazide": "Thiazide diuretic",
    "hctz": "Thiazide diuretic",
    "furosemide": "Loop diuretic",
    "lasix": "Loop diuretic",
    "spironolactone": "Potassium-sparing diuretic",
    "warfarin": "Anticoagulant (vitamin K antagonist)",
    "coumadin": "Anticoagulant (warfarin)",
    "apixaban": "Anticoagulant (DOAC)",
    "eliquis": "Anticoagulant (apixaban)",
    "rivaroxaban": "Anticoagulant (DOAC)",
    "xarelto": "Anticoagulant (rivaroxaban)",
    "clopidogrel": "Antiplatelet",
    "plavix": "Antiplatelet",
    "digoxin": "Cardiac glycoside",
    "amiodarone": "Antiarrhythmic",
    "nitroglycerin": "Nitrate vasodilator",
    "atorvastatin": "Statin",
    "lipitor": "Statin",
    "simvastatin": "Statin",
    "zocor": "Statin",
    "rosuvastatin": "Statin",
    "crestor": "Statin",
    "metformin": "Biguanide / antidiabetic",
    "glucophage": "Biguanide",
    "insulin": "Insulin therapy",
    "glipizide": "Sulfonylurea antidiabetic",
    "glimepiride": "Sulfonylurea antidiabetic",
    "sitagliptin": "DPP-4 inhibitor",
    "januvia": "DPP-4 inhibitor",
    "semaglutide": "GLP-1 receptor agonist",
    "ozempic": "GLP-1 agonist",
    "wegovy": "GLP-1 agonist",
    "empagliflozin": "SGLT-2 inhibitor",
    "jardiance": "SGLT-2 inhibitor",
    "albuterol": "SABA bronchodilator",
    "ventolin": "SABA bronchodilator",
    "salbutamol": "SABA bronchodilator",
    "fluticasone": "Inhaled corticosteroid",
    "advair": "ICS/LABA combination",
    "symbicort": "ICS/LABA combination",
    "montelukast": "Leukotriene receptor antagonist",
    "singulair": "LTRA",
    "tiotropium": "LAMA bronchodilator",
    "spiriva": "LAMA bronchodilator",
    "omeprazole": "Proton pump inhibitor",
    "prilosec": "PPI",
    "pantoprazole": "Proton pump inhibitor",
    "protonix": "PPI",
    "esomeprazole": "Proton pump inhibitor",
    "nexium": "PPI",
    "famotidine": "H2 blocker",
    "pepcid": "H2 blocker",
    "ondansetron": "Antiemetic",
    "zofran": "Antiemetic",
    "metoclopramide": "Prokinetic antiemetic",
    "loperamide": "Antidiarrheal",
    "imodium": "Antidiarrheal",
    "sertraline": "SSRI antidepressant",
    "zoloft": "SSRI",
    "fluoxetine": "SSRI antidepressant",
    "prozac": "SSRI",
    "escitalopram": "SSRI antidepressant",
    "lexapro": "SSRI",
    "citalopram": "SSRI antidepressant",
    "venlafaxine": "SNRI antidepressant",
    "effexor": "SNRI",
    "duloxetine": "SNRI antidepressant",
    "cymbalta": "SNRI",
    "bupropion": "NDRI antidepressant",
    "wellbutrin": "NDRI",
    "alprazolam": "Benzodiazepine",
    "xanax": "Benzodiazepine",
    "lorazepam": "Benzodiazepine",
    "ativan": "Benzodiazepine",
    "clonazepam": "Benzodiazepine",
    "klonopin": "Benzodiazepine",
    "diazepam": "Benzodiazepine",
    "valium": "Benzodiazepine",
    "zolpidem": "Sedative-hypnotic",
    "ambien": "Sedative-hypnotic",
    "quetiapine": "Atypical antipsychotic",
    "seroquel": "Atypical antipsychotic",
    "olanzapine": "Atypical antipsychotic",
    "risperidone": "Atypical antipsychotic",
    "lithium": "Mood stabilizer",
    "valproate": "Anticonvulsant / mood stabilizer",
    "lamotrigine": "Anticonvulsant / mood stabilizer",
    "levetiracetam": "Anticonvulsant",
    "topiramate": "Anticonvulsant",
    "levothyroxine": "Thyroid hormone replacement",
    "synthroid": "Thyroid hormone",
    "prednisone": "Systemic corticosteroid",
    "methylprednisolone": "Systemic corticosteroid",
    "dexamethasone": "Corticosteroid",
    "hydrocortisone": "Corticosteroid",
    "cetirizine": "H1 antihistamine",
    "zyrtec": "H1 antihistamine",
    "loratadine": "H1 antihistamine",
    "claritin": "H1 antihistamine",
    "diphenhydramine": "H1 antihistamine (sedating)",
    "benadryl": "H1 antihistamine",
    "fexofenadine": "H1 antihistamine",
    "allegra": "H1 antihistamine",
    "epinephrine": "Sympathomimetic / anaphylaxis",
    "epipen": "Epinephrine auto-injector",
    "sumatriptan": "Triptan / antimigraine",
    "imitrex": "Triptan",
    "rizatriptan": "Triptan / antimigraine",
    "vitamin d": "Vitamin D supplementation",
    "vitamin b12": "Vitamin B12",
    "folic acid": "Folate supplementation",
    "iron": "Iron supplementation",
    "calcium": "Calcium supplementation",
    "magnesium": "Magnesium supplementation",
    "fish oil": "Omega-3 supplement",
    "multivitamin": "Multivitamin",
}

DOSE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|mL|units?|IU|tablets?|tabs?|caps?|capsules?|puffs?|drops?)",
    re.I,
)
FREQ_RE = re.compile(
    r"\b(once|twice|three times|four times|every\s+\d+\s+hours?|"
    r"q\d+h|qd|bid|tid|qid|qhs|prn|daily|weekly|monthly|"
    r"every (?:morning|evening|night)|at bedtime|as needed)\b",
    re.I,
)
ROUTE_RE = re.compile(
    r"\b(orally?|po|iv|intravenously?|im|intramuscularly?|"
    r"subcutaneously?|sq|sc|topically?|inhaled?|sublingually?|"
    r"rectally?|transdermally?|intranasally?)\b",
    re.I,
)


# ═══════════════════════════════════════════════════════════════════════════
# Vital signs
# ═══════════════════════════════════════════════════════════════════════════


def _vital_builder_bp(m):
    s, d = int(m.group(1)), int(m.group(2))
    if 60 <= s <= 250 and 30 <= d <= 150:
        return Vital("Blood Pressure", f"{s}/{d}", "mmHg", m.group(0))
    return None


VITAL_PATTERNS = [
    # Spoken BP: "162 over 94"
    (
        re.compile(r"(?:bp|blood pressure)[:\s]+(\d{2,3})\s+over\s+(\d{2,3})", re.I),
        lambda m: Vital(
            "Blood Pressure", f"{m.group(1)}/{m.group(2)}", "mmHg", m.group(0)
        ),
    ),
    (
        re.compile(r"\b(\d{2,3})\s+over\s+(\d{2,3})\b", re.I),
        lambda m: Vital(
            "Blood Pressure", f"{m.group(1)}/{m.group(2)}", "mmHg", m.group(0)
        )
        if 60 <= int(m.group(1)) <= 250 and 30 <= int(m.group(2)) <= 150
        else None,
    ),
    (
        re.compile(r"(?:bp|blood pressure)[:\s]+(\d{2,3})\s*/\s*(\d{2,3})", re.I),
        lambda m: Vital(
            "Blood Pressure", f"{m.group(1)}/{m.group(2)}", "mmHg", m.group(0)
        ),
    ),
    (re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?\b", re.I), _vital_builder_bp),
    (
        re.compile(r"(?:hr|heart rate|pulse)[:\s]+(\d{2,3})\s*(?:bpm)?", re.I),
        lambda m: Vital("Heart Rate", m.group(1), "bpm", m.group(0)),
    ),
    (
        re.compile(r"\b(\d{2,3})\s*bpm\b", re.I),
        lambda m: Vital("Heart Rate", m.group(1), "bpm", m.group(0)),
    ),
    (
        re.compile(
            r"(?:temp(?:erature)?)[:\s]+(\d+(?:\.\d+)?)\s*(?:degrees?|deg)?(?:\s*[FC])?",
            re.I,
        ),
        lambda m: Vital("Temperature", m.group(1), "F", m.group(0)),
    ),
    (
        re.compile(r"\b(9[5-9]|10[0-5])\.\d\b"),
        lambda m: Vital("Temperature", m.group(0), "F", m.group(0)),
    ),
    (
        re.compile(
            r"(?:o2\s*sat(?:uration)?|spo2|oxygen\s+saturation)[:\s]+([\d.]+)\s*%?",
            re.I,
        ),
        lambda m: Vital("O2 Saturation", m.group(1), "%", m.group(0)),
    ),
    (
        re.compile(r"\b(9[0-9]|100)\s*%\s*(?:on\s+(?:room air|ra))?", re.I),
        lambda m: Vital("O2 Saturation", m.group(1), "%", m.group(0)),
    ),
    (
        re.compile(r"(?:rr|resp(?:iratory)?\s+rate)[:\s]+(\d{1,2})", re.I),
        lambda m: Vital("Respiratory Rate", m.group(1), "breaths/min", m.group(0)),
    ),
    (
        re.compile(r"(?:weight|wt)[:\s]+([\d.]+)\s*(kg|lbs?|pounds?)", re.I),
        lambda m: Vital("Weight", m.group(1), m.group(2), m.group(0)),
    ),
    (
        re.compile(r"(?:bmi)[:\s]+([\d.]+)", re.I),
        lambda m: Vital("BMI", m.group(1), "kg/m2", m.group(0)),
    ),
    (
        re.compile(r"(?:glucose|blood sugar|fbs)[:\s]+([\d.]+)\s*(?:mg/dl)?", re.I),
        lambda m: Vital("Blood Glucose", m.group(1), "mg/dL", m.group(0)),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# ICD-10 map
# ═══════════════════════════════════════════════════════════════════════════

ICD10_MAP = {
    "pneumonia": ("J18.9", "Pneumonia, unspecified"),
    "bronchitis": ("J40", "Bronchitis, unspecified"),
    "asthma": ("J45.909", "Unspecified asthma, uncomplicated"),
    "copd": ("J44.1", "COPD with acute exacerbation"),
    "upper respiratory infection": ("J06.9", "Acute upper respiratory infection"),
    "uri": ("J06.9", "Acute upper respiratory infection"),
    "sinusitis": ("J32.9", "Chronic sinusitis, unspecified"),
    "pharyngitis": ("J02.9", "Acute pharyngitis, unspecified"),
    "strep throat": ("J02.0", "Streptococcal pharyngitis"),
    "influenza": ("J11.1", "Influenza with respiratory manifestations"),
    "hypertension": ("I10", "Essential (primary) hypertension"),
    "heart failure": ("I50.9", "Heart failure, unspecified"),
    "atrial fibrillation": ("I48.91", "Unspecified atrial fibrillation"),
    "afib": ("I48.91", "Unspecified atrial fibrillation"),
    "coronary artery disease": ("I25.10", "Atherosclerotic heart disease"),
    "myocardial infarction": ("I21.9", "Acute myocardial infarction, unspecified"),
    "chest pain": ("R07.9", "Chest pain, unspecified"),
    "angina": ("I20.9", "Angina pectoris, unspecified"),
    "dvt": ("I82.409", "Deep vein thrombosis, unspecified"),
    "pulmonary embolism": ("I26.99", "Pulmonary embolism"),
    "diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "type 2 diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "type 1 diabetes": ("E10.9", "Type 1 diabetes mellitus without complications"),
    "hypothyroidism": ("E03.9", "Hypothyroidism, unspecified"),
    "hyperthyroidism": ("E05.90", "Thyrotoxicosis, unspecified"),
    "obesity": ("E66.9", "Obesity, unspecified"),
    "hyperlipidemia": ("E78.5", "Hyperlipidemia, unspecified"),
    "gerd": ("K21.0", "GERD with esophagitis"),
    "acid reflux": ("K21.9", "GERD without esophagitis"),
    "peptic ulcer": ("K27.9", "Peptic ulcer, unspecified"),
    "ibs": ("K58.9", "Irritable bowel syndrome"),
    "osteoarthritis": ("M19.90", "Primary osteoarthritis, unspecified site"),
    "rheumatoid arthritis": ("M06.9", "Rheumatoid arthritis, unspecified"),
    "gout": ("M10.9", "Gout, unspecified"),
    "back pain": ("M54.5", "Low back pain"),
    "low back pain": ("M54.5", "Low back pain"),
    "fibromyalgia": ("M79.7", "Fibromyalgia"),
    "osteoporosis": ("M81.0", "Age-related osteoporosis"),
    "migraine": ("G43.909", "Migraine, unspecified"),
    "headache": ("R51", "Headache"),
    "depression": ("F32.9", "Major depressive disorder, unspecified"),
    "anxiety": ("F41.9", "Anxiety disorder, unspecified"),
    "insomnia": ("G47.00", "Insomnia, unspecified"),
    "ptsd": ("F43.10", "Post-traumatic stress disorder, unspecified"),
    "uti": ("N39.0", "Urinary tract infection, unspecified"),
    "urinary tract infection": ("N39.0", "Urinary tract infection, unspecified"),
    "kidney stones": ("N20.0", "Calculus of kidney"),
    "orchialgia": ("N50.9", "Disorder of male genital organs, unspecified"),
    "testicular pain": ("N50.9", "Disorder of male genital organs, unspecified"),
    "epididymitis": ("N45.1", "Epididymitis"),
    "dysmenorrhea": ("N94.6", "Dysmenorrhea, unspecified"),
    "anemia": ("D64.9", "Anemia, unspecified"),
    "allergic rhinitis": ("J30.9", "Allergic rhinitis, unspecified"),
    "eczema": ("L30.9", "Dermatitis, unspecified"),
    "psoriasis": ("L40.9", "Psoriasis, unspecified"),
    "sleep apnea": ("G47.33", "Obstructive sleep apnea"),
    "cellulitis": ("L03.90", "Cellulitis, unspecified"),
    "sepsis": ("A41.9", "Sepsis, unspecified organism"),
}

PLAN_KEYWORDS = [
    "prescribe",
    "prescribed",
    "start ",
    "begin",
    "continue",
    "stop ",
    "discontinue",
    "refer",
    "referral",
    "order",
    "schedule",
    "return",
    "follow up",
    "follow-up",
    "monitor",
    "blood work",
    "lab",
    "labs",
    "x-ray",
    "xray",
    "mri",
    "ct scan",
    "ultrasound",
    "ecg",
    "ekg",
    "echo",
    "biopsy",
    "injection",
]

SOCIAL_PATTERNS = {
    "smoking": re.compile(
        r"\b(smok(?:es?|ing|er)|cigarette|tobacco|pack\s+(?:a|per)\s+(?:day|week))\b",
        re.I,
    ),
    "alcohol": re.compile(
        r"\b(alcohol|drink(?:ing|s)?|beer|wine|liquor|social drinker)\b", re.I
    ),
    "drugs": re.compile(
        r"\b(drug use|illicit|recreational|marijuana|cannabis|cocaine|heroin|meth)\b",
        re.I,
    ),
    "exercise": re.compile(
        r"\b(exercis(?:e|ing|es)|workout|gym|walking|running|sedentary)\b", re.I
    ),
    "occupation": re.compile(
        r"\b(works?\s+as|employed|occupation|retired|unemployed|disability)\b", re.I
    ),
    "marital_status": re.compile(
        r"\b(married|single|divorced|widowed|partner)\b", re.I
    ),
}

ALLERGY_PATTERNS = [
    re.compile(r"allerg(?:ic|y|ies)\s+to\s+([\w\s,]+?)(?:\.|,|;|\n|$)", re.I),
    re.compile(r"nkda|no known drug allergies|no known allergies", re.I),
]

FAMILY_HISTORY_TERMS = [
    "family history",
    "mother",
    "father",
    "parents",
    "sibling",
    "brother",
    "sister",
    "grandfather",
    "grandmother",
    "runs in the family",
    "familial",
    "hereditary",
]

SEVERITY_TERMS = {
    "mild": "mild",
    "slight": "mild",
    "minor": "mild",
    "a bit": "mild",
    "a little": "mild",
    "manageable": "mild",
    "moderate": "moderate",
    "significant": "moderate",
    "pretty bad": "moderate",
    "quite bad": "moderate",
    "bothering me": "moderate",
    "severe": "severe",
    "intense": "severe",
    "extreme": "severe",
    "excruciating": "severe",
    "unbearable": "severe",
    "terrible": "severe",
    "worst": "severe",
    "awful": "severe",
    "horrible": "severe",
    "really bad": "severe",
    "killing me": "severe",
    "kills me": "severe",
    "can't stand it": "severe",
    "agony": "severe",
    "very bad": "severe",
    "so bad": "severe",
    "bad": "moderate",
}

NEGATION_CUES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies?\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bnever\b",
    r"\bno history of\b",
    r"\babsent\b",
    r"\brules?\s+out\b",
]

# Entity labels per model type
# en_ner_bc5cdr_md: "DISEASE", "CHEMICAL"
# en_core_sci_*:   "ENTITY"
# en_core_web_sm:  no medical entities
SPACY_SYMPTOM_LABELS = {
    "DISEASE",
    "PROBLEM",
    "SYMPTOM",
    "SIGN_SYMPTOM",
    "FINDING",
    "ENTITY",
}
SPACY_MED_LABELS = {"CHEMICAL", "DRUG", "ENTITY"}


# ═══════════════════════════════════════════════════════════════════════════
# Main NLP engine
# ═══════════════════════════════════════════════════════════════════════════


class MedicalNLPEngine:
    """
    spaCy + scispaCy primary NER, enhanced by colloquial lexicon.
    Negation via sentence-boundary-aware rule matching.
    Gracefully degrades to pure rule-based if no spaCy model is installed.
    """

    def __init__(self):
        self.nlp, self.model_name = _load_spacy()

        # Pre-compile lexicon regex (longest match first)
        # Matches base form + common suffixes: s, es, ing, ed, er
        _terms = sorted(COLLOQUIAL_TO_CLINICAL.keys(), key=len, reverse=True)
        self._symptom_re = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in _terms) + r")(?:s|es|ing|ed|er)?\b", re.I
        )
        _meds = sorted(MEDICATION_TERMS.keys(), key=len, reverse=True)
        self._med_re = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in _meds) + r")\b", re.I
        )

        # Body-part pain pattern: catches "pain in my balls", "ache in my knee" etc.
        # that aren't covered by the exact-phrase colloquial dict
        self._body_pain_re = re.compile(
            r"\b(?:pain|ache|aching|hurt(?:s|ing)?|sore(?:ness)?|discomfort|burning|"
            r"swelling|swollen|tender(?:ness)?)\s+(?:in|around|near|at|on|of)?\s*"
            r"(?:my|his|her|the|both)?\s*"
            r"(balls?|nut(?:s)?|testicle(?:s)?|scrota(?:l)?|scrotum|groin|penis|penile|"
            r"vagina(?:l)?|vulva(?:r)?|uterus|womb|ovary|ovaries|ovarian|"
            r"knee(?:s)?|hip(?:s)?|shoulder(?:s)?|elbow(?:s)?|wrist(?:s)?|ankle(?:s)?|"
            r"shin(?:s)?|calf|calves|thigh(?:s)?|toe(?:s)?|finger(?:s)?|thumb(?:s)?|"
            r"jaw|ear(?:s)?|eye(?:s)?|temple(?:s)?|forehead|neck|throat)",
            re.I,
        )

        # Map body-part words -> clinical term
        self._body_part_map = {
            "balls": "Orchialgia / testicular pain",
            "ball": "Orchialgia / testicular pain",
            "nuts": "Orchialgia / testicular pain",
            "nut": "Orchialgia / testicular pain",
            "testicle": "Orchialgia / testicular pain",
            "testicles": "Orchialgia / testicular pain",
            "scrotum": "Scrotal pain",
            "scrotal": "Scrotal pain",
            "scrota": "Scrotal pain",
            "groin": "Groin pain",
            "penis": "Penile pain",
            "penile": "Penile pain",
            "vagina": "Vaginal pain / dyspareunia",
            "vaginal": "Vaginal pain / dyspareunia",
            "vulva": "Vulvar pain",
            "vulvar": "Vulvar pain",
            "uterus": "Uterine / pelvic pain",
            "womb": "Uterine / pelvic pain",
            "ovary": "Ovarian / pelvic pain",
            "ovaries": "Ovarian / pelvic pain",
            "ovarian": "Ovarian / pelvic pain",
            "knee": "Knee pain",
            "knees": "Knee pain",
            "shoulder": "Shoulder pain",
            "shoulders": "Shoulder pain",
            "elbow": "Elbow pain",
            "elbows": "Elbow pain",
            "wrist": "Wrist pain",
            "wrists": "Wrist pain",
            "hip": "Hip pain",
            "hips": "Hip pain",
            "ankle": "Ankle pain",
            "ankles": "Ankle pain",
            "shin": "Shin pain",
            "shins": "Shin pain",
            "calf": "Calf pain",
            "calves": "Calf pain",
            "thigh": "Thigh pain",
            "thighs": "Thigh pain",
            "jaw": "Jaw pain / TMJ",
            "ear": "Otalgia / ear pain",
            "ears": "Otalgia / ear pain",
            "eye": "Ocular pain",
            "eyes": "Ocular pain",
            "forehead": "Frontal headache / forehead pain",
            "temple": "Temporal headache",
            "temples": "Temporal headache",
            "throat": "Throat pain / pharyngitis",
            "finger": "Finger pain",
            "fingers": "Finger pain",
            "thumb": "Thumb pain",
            "toe": "Toe pain",
            "toes": "Toe pain",
        }

    # ── Negation ──────────────────────────────────────────────────────────

    def _is_negated(self, text: str, start: int) -> bool:
        """
        Sentence-boundary-aware negation.
        Only looks within the same sentence (does not cross . ? ! newline).
        """
        before = text[max(0, start - 80) : start]
        # Clip to last sentence boundary so negation from a previous sentence
        # doesn't bleed into the current one
        for sep in (".", "!", "?", "\n"):
            idx = before.rfind(sep)
            if idx != -1:
                before = before[idx + 1 :]
        before = before.lower()
        return any(re.search(p, before) for p in NEGATION_CUES)

    # ── Context helpers ───────────────────────────────────────────────────

    def _ctx(self, text: str, start: int, end: int, w: int = 120) -> str:
        return text[max(0, start - w) : min(len(text), end + w)]

    def _severity(self, ctx: str) -> Optional[str]:
        cl = ctx.lower()
        for term, sev in SEVERITY_TERMS.items():
            if term in cl:
                return sev
        m = re.search(r"(\d+)\s*(?:/|out of)\s*10", ctx, re.I)
        if m:
            n = int(m.group(1))
            return "mild" if n <= 3 else "severe" if n >= 7 else "moderate"
        return None

    def _duration(self, ctx: str) -> Optional[str]:
        for pat in [
            r"for\s+(?:the\s+(?:past|last)\s+)?(\d+\s*(?:day|days|week|weeks|month|months|year|years))",
            r"since\s+(\w+(?:\s+\w+)?)",
            r"(\d+\s*(?:day|days|week|weeks|month|months|year|years))\s*(?:ago|duration)",
            r"(yesterday|today|last\s+(?:week|month|night))",
        ]:
            m = re.search(pat, ctx, re.I)
            if m:
                return m.group(1)
        return None

    def _character(self, ctx: str) -> Optional[str]:
        chars = [
            "sharp",
            "dull",
            "burning",
            "stabbing",
            "throbbing",
            "aching",
            "cramping",
            "shooting",
            "radiating",
            "squeezing",
            "tearing",
            "gnawing",
            "constant",
            "intermittent",
            "colicky",
            "pressure-like",
        ]
        found = [c for c in chars if re.search(r"\b" + c + r"\b", ctx, re.I)]
        return ", ".join(found) if found else None

    def _location(self, ctx: str) -> Optional[str]:
        regions = [
            "right upper quadrant",
            "left upper quadrant",
            "right lower quadrant",
            "left lower quadrant",
            # Genitourinary — must be BEFORE generic "head" to prevent misclassification
            "testicle",
            "testicular",
            "scrotum",
            "scrotal",
            "groin",
            "pelvis",
            "pelvic",
            "perineum",
            "penis",
            "penile",
            "vagina",
            "vaginal",
            "vulva",
            "ovary",
            "ovarian",
            # General body regions
            "head",
            "neck",
            "throat",
            "chest",
            "abdomen",
            "back",
            "shoulder",
            "arm",
            "elbow",
            "wrist",
            "hand",
            "hip",
            "thigh",
            "leg",
            "knee",
            "ankle",
            "foot",
            "left",
            "right",
            "bilateral",
            "lumbar",
            "thoracic",
            "cervical",
            "epigastric",
        ]
        found = [
            r for r in regions if re.search(r"\b" + re.escape(r) + r"\b", ctx, re.I)
        ]
        return ", ".join(found[:3]) if found else None


    # ── Symptom extraction ────────────────────────────────────────────────

    def _spacy_symptoms(self, text: str) -> Tuple[List[Symptom], set]:
        """Extract symptoms via spaCy NER. Returns (symptoms, seen_canonicals)."""
        symptoms: List[Symptom] = []
        seen: set = set()
        if not self.nlp:
            return symptoms, seen
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ not in SPACY_SYMPTOM_LABELS:
                    continue
                # For ENTITY labels (general scispaCy), only keep if it maps
                # to our lexicon — avoids noisy generic matches
                name_lower = ent.text.strip().lower()
                if ent.label_ == "ENTITY":
                    canonical = COLLOQUIAL_TO_CLINICAL.get(name_lower)
                    if not canonical:
                        continue
                else:
                    canonical = COLLOQUIAL_TO_CLINICAL.get(
                        name_lower, ent.text.strip().title()
                    )
                if canonical in seen:
                    continue
                seen.add(canonical)
                ctx = self._ctx(text, ent.start_char, ent.end_char)
                symptoms.append(
                    Symptom(
                        name=canonical,
                        negated=self._is_negated(text, ent.start_char),
                        severity=self._severity(ctx),
                        duration=self._duration(ctx),
                        character=self._character(ctx),
                        location=self._location(ctx),
                        context=ctx.strip(),
                        source="spacy",
                    )
                )
        except Exception:
            pass
        return symptoms, seen

    def _lexicon_symptoms(self, text: str, already_seen: set) -> List[Symptom]:
        """Lexicon + colloquial pattern pass — catches what spaCy missed."""
        symptoms: List[Symptom] = []
        seen = set(already_seen)

        for m in self._symptom_re.finditer(text):
            raw = m.group(1).lower()  # group(1) = base term without plural suffix
            canonical = COLLOQUIAL_TO_CLINICAL.get(raw, raw.title())
            if canonical in seen:
                continue
            seen.add(canonical)
            ctx = self._ctx(text, m.start(), m.end())
            symptoms.append(
                Symptom(
                    name=canonical,
                    negated=self._is_negated(text, m.start()),
                    severity=self._severity(ctx),
                    duration=self._duration(ctx),
                    character=self._character(ctx),
                    location=self._location(ctx),
                    context=ctx.strip(),
                    source="rule",
                )
            )

        for pattern, canonical in COLLOQUIAL_PATTERNS:
            for m in pattern.finditer(text):
                if canonical in seen:
                    continue
                seen.add(canonical)
                ctx = self._ctx(text, m.start(), m.end())
                symptoms.append(
                    Symptom(
                        name=canonical,
                        negated=self._is_negated(text, m.start()),
                        severity=self._severity(ctx),
                        duration=self._duration(ctx),
                        character=self._character(ctx),
                        location=self._location(ctx),
                        context=ctx.strip(),
                        source="colloquial",
                    )
                )

        # Body-part pain pattern: catches "pain in my balls", "hurt in my knee" etc.
        for m in self._body_pain_re.finditer(text):
            part_word = m.group(1).lower()
            canonical = self._body_part_map.get(part_word)
            if not canonical:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            ctx = self._ctx(text, m.start(), m.end())
            symptoms.append(
                Symptom(
                    name=canonical,
                    negated=self._is_negated(text, m.start()),
                    severity=self._severity(ctx),
                    duration=self._duration(ctx),
                    character=self._character(ctx),
                    location=part_word,
                    context=ctx.strip(),
                    source="colloquial",
                )
            )

        return symptoms

    def _patient_lines(self, text: str) -> str:
        """
        Return only the patient-spoken lines from a transcript.
        Prevents doctor questions ("Any fever?") from triggering positive findings.
        Falls back to full text if no speaker labels are detected.
        """
        lines = text.split("\n")
        patient_lines = []
        in_patient_turn = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^(?:patient|pt)\s*:", stripped, re.I):
                in_patient_turn = True
                # Strip the speaker label
                content = re.sub(r"^(?:patient|pt)\s*:\s*", "", stripped, flags=re.I)
                if content:
                    patient_lines.append(content)
            elif re.match(
                r"^(?:doctor|dr|physician|md|np|pa|nurse)\s*:", stripped, re.I
            ):
                in_patient_turn = False
            elif in_patient_turn:
                # Continuation line of patient speech
                patient_lines.append(stripped)
            # Lines with no speaker label and not in a patient turn are skipped

        result = "\n".join(patient_lines).strip()
        # If we found no speaker-labelled patient turns, fall back to full text
        # (e.g. transcript was pasted without labels)
        return result if result else text

    def _doctor_lines(self, text: str) -> str:
        """
        Return only clinician-spoken lines from a transcript.
        Used to prioritise doctor-documented vitals over patient-reported ones.
        Falls back to full text if no speaker labels found.
        """
        lines = text.split("\n")
        doctor_lines = []
        in_doctor_turn = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^(?:doctor|dr|physician|md|np|pa|nurse)\s*:", stripped, re.I):
                in_doctor_turn = True
                content_line = re.sub(
                    r"^(?:doctor|dr|physician|md|np|pa|nurse)\s*:\s*",
                    "",
                    stripped,
                    flags=re.I,
                )
                if content_line:
                    doctor_lines.append(content_line)
            elif re.match(r"^(?:patient|pt)\s*:", stripped, re.I):
                in_doctor_turn = False
            elif in_doctor_turn:
                doctor_lines.append(stripped)

        result = "\n".join(doctor_lines).strip()
        return result if result else text

    def extract_symptoms(self, text: str) -> List[Symptom]:
        # Only extract symptoms from patient speech — prevents doctor questions
        # like "Any chest pain?" from generating false positive findings.
        patient_text = self._patient_lines(text)
        spacy_syms, seen = self._spacy_symptoms(patient_text)
        lexicon_syms = self._lexicon_symptoms(patient_text, seen)
        return spacy_syms + lexicon_syms

    # ── Medication extraction ─────────────────────────────────────────────

    def extract_medications(self, text: str) -> List[Medication]:
        seen: set = set()
        meds: List[Medication] = []

        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ not in SPACY_MED_LABELS:
                        continue
                    name_lower = ent.text.strip().lower()
                    if ent.label_ == "ENTITY":
                        if name_lower not in MEDICATION_TERMS:
                            continue
                    drug_class = MEDICATION_TERMS.get(name_lower, "")
                    canonical = (
                        f"{ent.text.strip().title()} ({drug_class})"
                        if drug_class
                        else ent.text.strip().title()
                    )
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    ctx = self._ctx(text, ent.start_char, ent.end_char, 150)
                    meds.append(self._build_med(canonical, ctx))
            except Exception:
                pass

        for m in self._med_re.finditer(text):
            raw = m.group(0).lower()
            drug_class = MEDICATION_TERMS.get(raw, "")
            canonical = (
                f"{m.group(0).title()} ({drug_class})"
                if drug_class
                else m.group(0).title()
            )
            if canonical in seen:
                continue
            seen.add(canonical)
            ctx = self._ctx(text, m.start(), m.end(), 150)
            meds.append(self._build_med(canonical, ctx))

        return meds

    def _build_med(self, name: str, ctx: str) -> Medication:
        status = "mentioned"
        cl = ctx.lower()
        if any(w in cl for w in ["prescribed", "started on", "start ", "initiate"]):
            status = "prescribed"
        elif any(
            w in cl for w in ["currently taking", "currently on", "takes ", "is on "]
        ):
            status = "current"
        elif any(w in cl for w in ["stopped", "discontinued", "no longer"]):
            status = "discontinued"
        elif any(w in cl for w in ["allergic to", "allergy to"]):
            status = "allergic"
        dose_m = DOSE_RE.search(ctx)
        freq_m = FREQ_RE.search(ctx)
        route_m = ROUTE_RE.search(ctx)
        return Medication(
            name=name,
            dose=f"{dose_m.group(1)} {dose_m.group(2)}" if dose_m else None,
            frequency=freq_m.group(0) if freq_m else None,
            route=route_m.group(0) if route_m else None,
            status=status,
        )

    # ── Vitals ────────────────────────────────────────────────────────────

    def _extract_vitals_from(self, text: str, seen: set) -> List[Vital]:
        """Extract vitals from a text segment, skipping names already in seen."""
        result: List[Vital] = []
        for pattern, builder in VITAL_PATTERNS:
            for m in pattern.finditer(text):
                try:
                    v = builder(m)
                    if v and v.name not in seen:
                        result.append(v)
                        seen.add(v.name)
                except Exception:
                    continue
        return result

    def extract_vitals(self, text: str) -> List[Vital]:
        """
        Extract vital signs, preferring clinician-documented values over
        patient-reported ones (e.g. doctor-measured BP takes priority over
        home BP mentioned by the patient).
        """
        seen: set = set()
        vitals: List[Vital] = []
        # First pass: doctor lines only
        doctor_text = self._doctor_lines(text)
        if doctor_text != text:  # speaker labels were present
            vitals += self._extract_vitals_from(doctor_text, seen)
        # Second pass: full text for anything the doctor lines missed
        vitals += self._extract_vitals_from(text, seen)
        return vitals

    # ── Diagnoses ─────────────────────────────────────────────────────────

    def extract_diagnoses(self, text: str) -> List[Diagnosis]:
        diagnoses: List[Diagnosis] = []
        seen: set = set()

        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ not in ("DISEASE", "ENTITY"):
                        continue
                    name_l = ent.text.strip().lower()
                    for condition, (code, desc) in ICD10_MAP.items():
                        if condition in name_l and condition not in seen:
                            seen.add(condition)
                            ctx = self._ctx(text, ent.start_char, ent.end_char, 60)
                            certainty = (
                                "possible"
                                if re.search(
                                    r"rule\s+out|r/o|possible|probable|suspect",
                                    ctx,
                                    re.I,
                                )
                                else "confirmed"
                            )
                            diagnoses.append(
                                Diagnosis(
                                    name=desc,
                                    icd10=code,
                                    certainty=certainty,
                                    primary=(len(diagnoses) == 0),
                                )
                            )
            except Exception:
                pass

        dx_patterns = [
            re.compile(
                r"(?:diagnos(?:ed?|is)|impression|assessment)[:\s]+([^.\n]+)", re.I
            ),
            re.compile(
                r"(?:consistent with|likely|history of|hx of)\s+([^.,\n]+)", re.I
            ),
        ]
        for pat in dx_patterns:
            for m in pat.finditer(text):
                fragment = m.group(1).strip().lower()
                for condition, (code, desc) in ICD10_MAP.items():
                    if condition in fragment and condition not in seen:
                        seen.add(condition)
                        diagnoses.append(
                            Diagnosis(
                                name=desc,
                                icd10=code,
                                certainty="possible",
                                primary=(len(diagnoses) == 0),
                            )
                        )
        return diagnoses[:10]

    # ── Other extractions ─────────────────────────────────────────────────

    def extract_allergies(self, text: str) -> List[str]:
        allergies: List[str] = []
        for pat in ALLERGY_PATTERNS:
            for m in pat.finditer(text):
                raw = m.group(0).strip()
                if re.search(r"nkda|no known", raw, re.I):
                    return ["NKDA – No Known Drug Allergies"]
                if m.lastindex and m.group(1):
                    for s in re.split(r",|and", m.group(1)):
                        s = s.strip()
                        if len(s) > 2:
                            allergies.append(s.title())
        return list(set(allergies))[:10]

    def extract_social_history(self, text: str) -> dict:
        social = {}
        for category, pat in SOCIAL_PATTERNS.items():
            m = pat.search(text)
            if m:
                ctx = self._ctx(text, m.start(), m.end(), 60)
                social[category] = ctx.strip()
        return social

    def extract_family_history(self, text: str) -> List[str]:
        items: List[str] = []
        for term in FAMILY_HISTORY_TERMS:
            for m in re.finditer(re.escape(term), text, re.I):
                end = min(len(text), m.end() + 120)
                snippet = text[m.start() : end]
                se = re.search(r"[.!?\n]", snippet)
                items.append(snippet[: se.start()] if se else snippet)
                if len(items) >= 5:
                    return items
        return items

    def extract_plan_items(self, text: str) -> List[str]:
        items: List[str] = []
        for sent in re.split(r"[.!?]\s+", text):
            sl = sent.lower()
            if any(kw in sl for kw in PLAN_KEYWORDS):
                items.append(sent.strip())
        return list(dict.fromkeys(items))[:15]

    def extract_review_of_systems(self, text: str) -> dict:
        # Use patient speech only so doctor questions don't create false positives
        text = self._patient_lines(text)
        systems = {
            "Constitutional": [
                "fever",
                "chills",
                "weight loss",
                "fatigue",
                "night sweats",
            ],
            "HEENT": [
                "headache",
                "blurred vision",
                "hearing loss",
                "tinnitus",
                "sore throat",
            ],
            "Respiratory": ["cough", "shortness of breath", "wheezing", "hemoptysis"],
            "Cardiovascular": ["chest pain", "palpitations", "edema", "leg swelling"],
            "Gastrointestinal": [
                "nausea",
                "vomiting",
                "diarrhea",
                "constipation",
                "heartburn",
                "abdominal pain",
            ],
            "Genitourinary": [
                "frequent urination",
                "painful urination",
                "blood in urine",
            ],
            "Musculoskeletal": ["joint pain", "back pain", "muscle pain", "stiffness"],
            "Neurological": [
                "dizziness",
                "numbness",
                "tingling",
                "weakness",
                "confusion",
            ],
            "Psychiatric": ["anxiety", "depression", "insomnia", "mood swings"],
            "Skin": ["rash", "itching", "bruising"],
        }
        ros = {}
        for system, syms in systems.items():
            positive, negative = [], []
            for s in syms:
                if re.search(r"\b" + re.escape(s) + r"\b", text, re.I):
                    idx = text.lower().find(s)
                    if idx != -1 and self._is_negated(text, idx):
                        negative.append(s)
                    else:
                        positive.append(s)
            if positive or negative:
                ros[system] = {"positive": positive, "negative": negative}
        return ros

    # ── Main entry ────────────────────────────────────────────────────────

    def analyze(self, text: str) -> ClinicalEntities:
        return ClinicalEntities(
            symptoms=self.extract_symptoms(text),
            medications=self.extract_medications(text),
            vitals=self.extract_vitals(text),
            diagnoses=self.extract_diagnoses(text),
            allergies=self.extract_allergies(text),
            family_history=self.extract_family_history(text),
            social_history=self.extract_social_history(text),
            review_of_systems=self.extract_review_of_systems(text),
            plan_items=self.extract_plan_items(text),
        )
