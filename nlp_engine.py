"""
Medical NLP Engine
Rule-based clinical entity extraction: symptoms, medications, vitals,
diagnoses, ICD-10 codes, negation detection, temporal context.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────── Data structures ────────────────────────────────

@dataclass
class Symptom:
    name: str
    negated: bool = False
    severity: Optional[str] = None
    duration: Optional[str] = None
    frequency: Optional[str] = None
    location: Optional[str] = None
    character: Optional[str] = None   # sharp, dull, burning, etc.
    context: str = ""

@dataclass
class Medication:
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    status: str = "mentioned"         # current / prescribed / discontinued / allergic

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
    certainty: str = "possible"       # confirmed / possible / ruled-out
    primary: bool = False

@dataclass
class ClinicalEntities:
    symptoms: list[Symptom] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    vitals: list[Vital] = field(default_factory=list)
    diagnoses: list[Diagnosis] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    family_history: list[str] = field(default_factory=list)
    social_history: dict = field(default_factory=dict)
    review_of_systems: dict = field(default_factory=dict)
    assessment_notes: list[str] = field(default_factory=list)
    plan_items: list[str] = field(default_factory=list)


# ─────────────────────────── Symptom lexicon ────────────────────────────────

SYMPTOM_TERMS = {
    # ── Pain / discomfort (clinical) ──────────────────────────────────────
    "headache": "Cephalgia",
    "migraine": "Migraine",
    "chest pain": "Chest pain",
    "chest tightness": "Chest tightness",
    "back pain": "Dorsalgia",
    "neck pain": "Cervicalgia",
    "abdominal pain": "Abdominal pain",
    "stomach pain": "Abdominal pain",
    "belly pain": "Abdominal pain",
    "joint pain": "Arthralgia",
    "muscle pain": "Myalgia",
    "muscle aches": "Myalgia",
    "sore throat": "Pharyngitis / sore throat",
    "ear pain": "Otalgia",
    "earache": "Otalgia",
    "eye pain": "Ophthalmalgia",
    "pelvic pain": "Pelvic pain",
    "flank pain": "Flank pain",
    "leg pain": "Lower extremity pain",
    "arm pain": "Upper extremity pain",
    "shoulder pain": "Shoulder pain",
    "knee pain": "Knee pain",
    "hip pain": "Hip pain",
    "wrist pain": "Wrist pain",
    "ankle pain": "Ankle pain",
    "foot pain": "Podialgia",
    "tooth pain": "Dentalgia / toothache",
    "toothache": "Dentalgia / toothache",
    "jaw pain": "Temporomandibular pain",
    "face pain": "Facial pain",
    "facial pain": "Facial pain",
    "groin pain": "Inguinal / groin pain",
    "testicular pain": "Orchialgia / testicular pain",
    "scrotal pain": "Scrotal pain",
    "penile pain": "Penile pain",
    "vaginal pain": "Vaginal pain",
    "vulvar pain": "Vulvodynia",
    "rectal pain": "Rectal pain / proctalgia",
    "anal pain": "Rectal pain / proctalgia",
    "pain": "Pain (unspecified)",

    # ── Colloquial pain / body part descriptions ──────────────────────────
    # Head
    "head hurts": "Cephalgia",
    "head is killing me": "Severe cephalgia",
    "head is pounding": "Cephalgia (pulsating)",
    "my head": "Cephalgia",
    "splitting headache": "Severe cephalgia",
    "banging headache": "Cephalgia",
    "throbbing head": "Cephalgia (pulsating)",

    # Throat / mouth
    "scratchy throat": "Pharyngitis / sore throat",
    "throat hurts": "Pharyngitis / sore throat",
    "hard to swallow": "Dysphagia",
    "can't swallow": "Dysphagia",
    "lump in throat": "Globus sensation / dysphagia",
    "mouth sores": "Oral ulceration",
    "mouth ulcers": "Oral ulceration",

    # Chest / heart
    "heart hurts": "Chest pain / cardiac",
    "heart is racing": "Tachycardia / palpitations",
    "heart pounding": "Palpitations",
    "heart beating fast": "Tachycardia / palpitations",
    "heart skipping": "Palpitations / dysrhythmia",
    "heart fluttering": "Palpitations",
    "tight chest": "Chest tightness",
    "pressure in chest": "Chest pressure",
    "can't catch my breath": "Dyspnea",
    "out of breath": "Dyspnea",
    "winded": "Exertional dyspnea",
    "can't breathe": "Dyspnea",
    "trouble breathing": "Dyspnea",

    # GI / stomach (colloquial)
    "tummy ache": "Abdominal pain",
    "tummy hurts": "Abdominal pain",
    "stomach ache": "Abdominal pain",
    "stomach is killing me": "Severe abdominal pain",
    "stomach cramps": "Abdominal cramping",
    "gut pain": "Abdominal pain",
    "belly ache": "Abdominal pain",
    "throwing up": "Vomiting",
    "puking": "Vomiting",
    "been sick": "Vomiting / nausea",
    "vomited": "Vomiting",
    "can't keep food down": "Vomiting / nausea",
    "feel like throwing up": "Nausea",
    "feel like vomiting": "Nausea",
    "queasy": "Nausea",
    "upset stomach": "Dyspepsia / nausea",
    "stomach is off": "Dyspepsia",
    "loose stool": "Diarrhea",
    "loose stools": "Diarrhea",
    "watery stool": "Diarrhea",
    "runny stool": "Diarrhea",
    "runs": "Diarrhea",
    "the runs": "Diarrhea",
    "pooping a lot": "Diarrhea",
    "can't poop": "Constipation",
    "haven't pooped": "Constipation",
    "blood in poop": "Hematochezia",
    "blood when i poop": "Hematochezia",
    "blood in my poop": "Hematochezia",
    "black poop": "Melena",
    "tarry stool": "Melena",
    "gassy": "Flatulence",
    "farting a lot": "Flatulence",
    "burping": "Eructation / belching",
    "burping a lot": "Eructation / belching",
    "bloated": "Abdominal distension / bloating",
    "feel bloated": "Abdominal distension / bloating",
    "stomach is bloated": "Abdominal distension / bloating",
    "not hungry": "Decreased appetite",
    "don't feel like eating": "Decreased appetite / anorexia",
    "can't eat": "Decreased appetite / anorexia",
    "no appetite": "Anorexia",
    "acid in my throat": "Gastroesophageal reflux",
    "stomach coming up": "Gastroesophageal reflux",

    # GU / genital (colloquial — full coverage)
    "balls hurt": "Orchialgia / testicular pain",
    "balls are hurting": "Orchialgia / testicular pain",
    "balls hurting": "Orchialgia / testicular pain",
    "ball pain": "Orchialgia / testicular pain",
    "testicle pain": "Orchialgia / testicular pain",
    "testicles hurt": "Orchialgia / testicular pain",
    "testicle hurts": "Orchialgia / testicular pain",
    "nuts hurt": "Orchialgia / testicular pain",
    "nut pain": "Orchialgia / testicular pain",
    "scrotal swelling": "Scrotal swelling",
    "swollen balls": "Scrotal swelling",
    "swollen testicle": "Testicular swelling",
    "dick hurts": "Penile pain",
    "penis hurts": "Penile pain",
    "penis pain": "Penile pain",
    "tip hurts": "Urethral / penile discomfort",
    "burning down there": "Dysuria / urogenital burning",
    "discharge from penis": "Urethral discharge",
    "penile discharge": "Urethral discharge",
    "vagina hurts": "Vaginal pain",
    "vaginal discharge": "Vaginal discharge",
    "discharge down there": "Vaginal / urethral discharge",
    "itchy down there": "Genital pruritus",
    "burning down below": "Dysuria / urogenital burning",
    "private parts hurt": "Genital / pelvic pain",
    "down there hurts": "Genital / pelvic pain",
    "groin hurts": "Inguinal / groin pain",
    "period pain": "Dysmenorrhea",
    "period cramps": "Dysmenorrhea",
    "painful periods": "Dysmenorrhea",
    "heavy periods": "Menorrhagia",
    "heavy bleeding": "Menorrhagia / hemorrhage",
    "irregular periods": "Menstrual irregularity",
    "missed period": "Amenorrhea",
    "no period": "Amenorrhea",
    "spotting": "Intermenstrual bleeding / spotting",
    "bleeding between periods": "Intermenstrual bleeding",
    "burning when i pee": "Dysuria",
    "burning when peeing": "Dysuria",
    "it burns when i pee": "Dysuria",
    "hurts to pee": "Dysuria",
    "pain when peeing": "Dysuria",
    "peeing a lot": "Urinary frequency / pollakiuria",
    "pee a lot": "Urinary frequency / pollakiuria",
    "going to the bathroom a lot": "Urinary frequency / pollakiuria",
    "up at night to pee": "Nocturia",
    "waking up to pee": "Nocturia",
    "can't hold my pee": "Urinary incontinence",
    "leaking urine": "Urinary incontinence",
    "blood in pee": "Hematuria",
    "blood when i pee": "Hematuria",
    "red urine": "Hematuria",
    "pink urine": "Hematuria",
    "cloudy urine": "Pyuria / cloudy urine",
    "smelly urine": "Malodorous urine",

    # Respiratory (colloquial)
    "cough": "Cough",
    "dry cough": "Dry cough",
    "productive cough": "Productive cough",
    "wet cough": "Productive cough",
    "coughing up stuff": "Productive cough",
    "coughing up mucus": "Productive cough with sputum",
    "coughing up phlegm": "Productive cough with sputum",
    "coughing up blood": "Hemoptysis",
    "shortness of breath": "Dyspnea",
    "short of breath": "Dyspnea",
    "difficulty breathing": "Dyspnea",
    "breathlessness": "Dyspnea",
    "wheezing": "Wheezing",
    "runny nose": "Rhinorrhea",
    "nose is running": "Rhinorrhea",
    "nasal congestion": "Nasal congestion",
    "stuffy nose": "Nasal congestion",
    "blocked nose": "Nasal congestion",
    "can't breathe through nose": "Nasal congestion",
    "sneezing": "Sneezing",
    "sneezing a lot": "Sneezing / rhinitis",
    "postnasal drip": "Postnasal drip",
    "mucus dripping": "Postnasal drip",
    "hoarseness": "Dysphonia",
    "lost my voice": "Dysphonia / aphonia",
    "voice is hoarse": "Dysphonia",
    "hemoptysis": "Hemoptysis",
    "coughing blood": "Hemoptysis",

    # Neuro (clinical + colloquial)
    "dizziness": "Dizziness",
    "dizzy": "Dizziness",
    "feel dizzy": "Dizziness",
    "feeling dizzy": "Dizziness",
    "room spinning": "Vertigo",
    "world is spinning": "Vertigo",
    "everything is spinning": "Vertigo",
    "vertigo": "Vertigo",
    "fainting": "Syncope",
    "fainted": "Syncope",
    "passed out": "Syncope",
    "blacked out": "Syncope",
    "nearly fainted": "Pre-syncope",
    "almost fainted": "Pre-syncope",
    "lightheadedness": "Pre-syncope / lightheadedness",
    "lightheaded": "Pre-syncope / lightheadedness",
    "feel faint": "Pre-syncope",
    "numbness": "Numbness / paresthesia",
    "numb": "Numbness / paresthesia",
    "pins and needles": "Paresthesia",
    "tingling": "Paresthesia",
    "weakness": "Weakness",
    "weak": "Weakness",
    "tremor": "Tremor",
    "shaking": "Tremor",
    "hands shaking": "Tremor",
    "seizure": "Seizure",
    "fit": "Seizure",
    "blackout": "Seizure / syncope",
    "confusion": "Confusion / altered mentation",
    "confused": "Confusion / altered mentation",
    "out of it": "Altered mentation / confusion",
    "not with it": "Altered mentation",
    "brain fog": "Cognitive impairment / brain fog",
    "can't think straight": "Cognitive impairment",
    "foggy brain": "Cognitive impairment / brain fog",
    "memory loss": "Amnesia / memory impairment",
    "forgetfulness": "Memory impairment",
    "forgetting things": "Memory impairment",
    "can't remember": "Memory impairment",
    "blurred vision": "Blurred vision",
    "blurry vision": "Blurred vision",
    "vision is blurry": "Blurred vision",
    "double vision": "Diplopia",
    "seeing double": "Diplopia",
    "vision loss": "Visual disturbance",
    "can't see properly": "Visual disturbance",
    "tinnitus": "Tinnitus",
    "ringing in ears": "Tinnitus",
    "ringing in my ears": "Tinnitus",
    "ears ringing": "Tinnitus",
    "hearing loss": "Hearing loss",
    "can't hear": "Hearing loss",
    "hard of hearing": "Hearing loss",
    "muffled hearing": "Hearing loss",

    # Constitutional (clinical + colloquial)
    "fever": "Pyrexia / fever",
    "high temperature": "Pyrexia / fever",
    "running a temperature": "Pyrexia / fever",
    "running a fever": "Pyrexia / fever",
    "temperature": "Pyrexia / fever",
    "chills": "Chills / rigors",
    "shivering": "Chills / rigors",
    "cold and shivery": "Chills / rigors",
    "night sweats": "Night sweats",
    "sweating at night": "Night sweats",
    "waking up sweating": "Night sweats",
    "drenched in sweat": "Diaphoresis / night sweats",
    "sweating a lot": "Diaphoresis",
    "fatigue": "Fatigue",
    "tiredness": "Fatigue",
    "exhaustion": "Fatigue / exhaustion",
    "tired": "Fatigue",
    "exhausted": "Fatigue / exhaustion",
    "wiped out": "Fatigue / exhaustion",
    "drained": "Fatigue",
    "run down": "Fatigue / malaise",
    "no energy": "Fatigue / lethargy",
    "can't get out of bed": "Debilitating fatigue",
    "always tired": "Chronic fatigue",
    "malaise": "Malaise",
    "feel terrible": "Malaise / general unwellness",
    "feel awful": "Malaise",
    "feel horrible": "Malaise",
    "feel unwell": "Malaise / general unwellness",
    "under the weather": "Malaise / general unwellness",
    "not feeling well": "Malaise / general unwellness",
    "feeling off": "Malaise",
    "weight loss": "Unintentional weight loss",
    "losing weight": "Unintentional weight loss",
    "losing weight without trying": "Unintentional weight loss",
    "weight gain": "Weight gain",
    "putting on weight": "Weight gain",
    "swelling": "Edema / swelling",
    "swollen": "Edema / swelling",
    "puffy": "Edema",
    "puffy legs": "Lower extremity edema",
    "puffy ankles": "Ankle edema",
    "puffy feet": "Pedal edema",
    "edema": "Edema",

    # Cardiac / vascular (clinical + colloquial)
    "palpitations": "Palpitations",
    "racing heart": "Tachycardia / palpitations",
    "irregular heartbeat": "Dysrhythmia",
    "chest pressure": "Chest pressure",
    "leg swelling": "Lower extremity edema",
    "ankle swelling": "Ankle edema",
    "cold hands": "Peripheral vasoconstriction / Raynaud's",
    "cold feet": "Peripheral vascular insufficiency",
    "legs going numb": "Lower extremity numbness",
    "leg cramps": "Lower extremity cramps / claudication",
    "calf pain": "Calf pain / possible DVT",
    "calf cramping": "Calf cramping / claudication",

    # GU / renal (clinical)
    "frequent urination": "Urinary frequency / pollakiuria",
    "painful urination": "Dysuria",
    "burning urination": "Dysuria",
    "blood in urine": "Hematuria",
    "urinary urgency": "Urinary urgency",
    "incontinence": "Urinary incontinence",
    "decreased urine output": "Oliguria",
    "nocturia": "Nocturia",
    "erectile dysfunction": "Erectile dysfunction",
    "can't get an erection": "Erectile dysfunction",
    "impotence": "Erectile dysfunction",

    # MSK / skin (clinical + colloquial)
    "rash": "Skin rash / dermatitis",
    "skin rash": "Skin rash / dermatitis",
    "breakout": "Skin rash / acneiform eruption",
    "spots": "Skin lesion / acneiform eruption",
    "spotty": "Skin rash / acneiform eruption",
    "hives": "Urticaria",
    "itching": "Pruritus",
    "itchy": "Pruritus",
    "itchy skin": "Pruritus",
    "scratching": "Pruritus",
    "skin lesion": "Skin lesion",
    "bruising": "Ecchymosis / bruising",
    "bruises easily": "Easy bruising / ecchymosis",
    "hair loss": "Alopecia",
    "losing hair": "Alopecia",
    "hair falling out": "Alopecia",
    "stiffness": "Joint stiffness",
    "stiff": "Joint stiffness",
    "morning stiffness": "Morning stiffness",
    "stiff in the morning": "Morning stiffness",
    "swollen joints": "Joint swelling / synovitis",
    "joints are swollen": "Joint swelling / synovitis",
    "joint swelling": "Joint swelling / synovitis",
    "muscle weakness": "Myasthenia / muscle weakness",
    "muscles are weak": "Muscle weakness",
    "back is killing me": "Severe dorsalgia",
    "back is really bad": "Dorsalgia",
    "can't move my back": "Severe dorsalgia",
    "knee is killing me": "Severe knee pain",
    "can't bend my knee": "Knee pain / limited range of motion",
    "clicking knee": "Knee crepitus",
    "clicking joints": "Joint crepitus",
    "lump": "Mass / lump (unspecified)",
    "lump under skin": "Subcutaneous mass",
    "lump in neck": "Cervical mass / lymphadenopathy",
    "swollen glands": "Lymphadenopathy",
    "swollen lymph nodes": "Lymphadenopathy",
    "wound not healing": "Impaired wound healing",

    # Psych / mental health (clinical + colloquial)
    "anxiety": "Anxiety",
    "anxious": "Anxiety",
    "nervous": "Anxiety / nervousness",
    "on edge": "Anxiety",
    "worrying a lot": "Anxiety / excessive worry",
    "can't stop worrying": "Anxiety / excessive worry",
    "freaking out": "Anxiety / panic",
    "depression": "Depressive symptoms",
    "depressed": "Depressive symptoms",
    "feeling low": "Depressive symptoms",
    "feeling down": "Depressive symptoms",
    "feel hopeless": "Depressive symptoms",
    "feel worthless": "Depressive symptoms",
    "no motivation": "Depressive symptoms / amotivation",
    "can't enjoy things": "Anhedonia",
    "lost interest": "Anhedonia",
    "insomnia": "Insomnia",
    "sleep problems": "Sleep disturbance",
    "can't sleep": "Insomnia",
    "trouble sleeping": "Insomnia",
    "waking up at night": "Sleep disturbance / insomnia",
    "can't stay asleep": "Sleep maintenance insomnia",
    "sleeping too much": "Hypersomnia",
    "mood changes": "Mood disturbance",
    "mood swings": "Mood disturbance / emotional lability",
    "irritability": "Irritability",
    "irritable": "Irritability",
    "easily irritated": "Irritability",
    "snapping at people": "Irritability",
    "panic attacks": "Panic attacks",
    "panic attack": "Panic attacks",
    "feeling like i'm going to die": "Panic attacks / anxiety",
    "losing my mind": "Anxiety / dissociation",
    "going crazy": "Anxiety / dissociation",
    "stress": "Psychosocial stress",
    "stressed": "Psychosocial stress",
    "stressed out": "Psychosocial stress",
    "burnt out": "Burnout / chronic stress",
    "burnout": "Burnout / chronic stress",
    "suicidal thoughts": "Suicidal ideation",
    "thinking about suicide": "Suicidal ideation",
    "self harm": "Self-harm",
    "hurting myself": "Self-harm",
    "hearing voices": "Auditory hallucinations",
    "seeing things": "Visual hallucinations",

    # Endocrine (clinical + colloquial)
    "excessive thirst": "Polydipsia",
    "increased thirst": "Polydipsia",
    "really thirsty": "Polydipsia",
    "always thirsty": "Polydipsia",
    "drinking a lot": "Polydipsia",
    "excessive urination": "Polyuria",
    "hot flashes": "Hot flashes / vasomotor symptoms",
    "hot flush": "Hot flashes / vasomotor symptoms",
    "hot flushes": "Hot flashes / vasomotor symptoms",
    "cold intolerance": "Cold intolerance",
    "always cold": "Cold intolerance",
    "heat intolerance": "Heat intolerance",
    "always hot": "Heat intolerance",
    "feeling shaky": "Tremor / hypoglycemia",
    "shaky": "Tremor / possible hypoglycemia",
    "feel shaky": "Tremor / possible hypoglycemia",
    "low blood sugar feeling": "Hypoglycemia symptoms",

    # Eyes / ENT (colloquial)
    "red eyes": "Conjunctival injection / red eye",
    "pink eye": "Conjunctivitis",
    "watery eyes": "Epiphora / lacrimation",
    "eyes watering": "Epiphora / lacrimation",
    "eye is swollen": "Periorbital edema",
    "eye discharge": "Ocular discharge",
    "crusty eyes": "Ocular discharge / blepharitis",
    "sore eyes": "Ocular discomfort",
    "ear fullness": "Ear fullness / Eustachian tube dysfunction",
    "blocked ears": "Ear fullness / Eustachian tube dysfunction",
    "ear discharge": "Otorrhea",
    "ear ringing": "Tinnitus",
    "ear bleeding": "Otorrhagia",
    "nosebleed": "Epistaxis",
    "nose is bleeding": "Epistaxis",
    "blood from nose": "Epistaxis",
    "post nasal drip": "Postnasal drip",
    "snoring": "Snoring / possible sleep apnea",
    "grinding teeth": "Bruxism",
}

# ─────────────────────────── Dynamic colloquial phrase patterns ──────────────
# These catch "my X hurts / is killing me / is swollen" style sentences
# that static keyword matching would miss entirely.

_PAIN_VERBS = r"(?:hurt(?:s|ing)?|ache?s?|aching|is killing me|kill(?:s|ing) me|is sore|is tender|is throbbing|is burning|burning|is painful|in pain)"
_SWELLING_VERBS = r"(?:is swollen|(?:are|is) swoll?en|has swollen|swelled up|swelling)"

COLLOQUIAL_BODY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Testicular / scrotal
    (re.compile(r"\b(?:my\s+)?(?:balls?|testicles?|nuts?|scrotum)\s*" + _PAIN_VERBS, re.I),
     "Orchialgia / testicular pain"),
    (re.compile(r"\b(?:my\s+)?(?:balls?|testicles?|nuts?)\s*" + _SWELLING_VERBS, re.I),
     "Testicular / scrotal swelling"),

    # Penile
    (re.compile(r"\b(?:my\s+)?(?:dick|penis|cock|willy|todger|member)\s*" + _PAIN_VERBS, re.I),
     "Penile pain"),
    (re.compile(r"\b(?:discharge|dripping|leaking)\s+from\s+(?:my\s+)?(?:dick|penis|willy)", re.I),
     "Urethral discharge"),

    # Vaginal / female GU
    (re.compile(r"\b(?:my\s+)?(?:vagina|vulva|vag|cooch|privates?)\s*" + _PAIN_VERBS, re.I),
     "Vaginal / vulvar pain"),
    (re.compile(r"\b(?:down\s+there|down\s+below|in\s+my\s+(?:private\s+area|groin))\s*" + _PAIN_VERBS, re.I),
     "Genital / pelvic pain"),

    # Stomach / abdomen
    (re.compile(r"\b(?:my\s+)?(?:stomach|tummy|belly|gut|abdomen)\s*" + _PAIN_VERBS, re.I),
     "Abdominal pain"),
    (re.compile(r"\b(?:my\s+)?(?:stomach|tummy|belly)\s*(?:is\s+)?(?:bloated|swollen|distended|big|huge)", re.I),
     "Abdominal distension / bloating"),

    # Back
    (re.compile(r"\b(?:my\s+)?(?:back|spine|lower back|upper back|lumbar)\s*" + _PAIN_VERBS, re.I),
     "Dorsalgia"),

    # Chest
    (re.compile(r"\b(?:my\s+)?(?:chest|breast bone|sternum|ribs?)\s*" + _PAIN_VERBS, re.I),
     "Chest pain"),

    # Head
    (re.compile(r"\b(?:my\s+)?(?:head|skull)\s*" + _PAIN_VERBS, re.I),
     "Cephalgia"),

    # Neck
    (re.compile(r"\b(?:my\s+)?(?:neck|throat area)\s*" + _PAIN_VERBS, re.I),
     "Cervicalgia / neck pain"),

    # Joints (general)
    (re.compile(r"\b(?:my\s+)?(?:knee|knees)\s*" + _PAIN_VERBS, re.I),
     "Knee pain"),
    (re.compile(r"\b(?:my\s+)?(?:hip|hips)\s*" + _PAIN_VERBS, re.I),
     "Hip pain"),
    (re.compile(r"\b(?:my\s+)?(?:shoulder|shoulders)\s*" + _PAIN_VERBS, re.I),
     "Shoulder pain"),
    (re.compile(r"\b(?:my\s+)?(?:ankle|ankles)\s*" + _PAIN_VERBS, re.I),
     "Ankle pain"),
    (re.compile(r"\b(?:my\s+)?(?:wrist|wrists)\s*" + _PAIN_VERBS, re.I),
     "Wrist pain"),
    (re.compile(r"\b(?:my\s+)?(?:elbow|elbows)\s*" + _PAIN_VERBS, re.I),
     "Elbow pain / epicondylalgia"),
    (re.compile(r"\b(?:my\s+)?(?:foot|feet|toe|toes)\s*" + _PAIN_VERBS, re.I),
     "Foot / toe pain"),
    (re.compile(r"\b(?:my\s+)?(?:finger|fingers|thumb|thumbs)\s*" + _PAIN_VERBS, re.I),
     "Hand / finger pain"),

    # Urination
    (re.compile(r"\b(?:it\s+)?(?:hurts?|burns?|stings?)\s+when\s+(?:i\s+)?(?:pee|urinate|go\s+to\s+the\s+bathroom)", re.I),
     "Dysuria"),
    (re.compile(r"\b(?:burning|pain|discomfort)\s+(?:when|while|during)\s+(?:peeing|urinating|urination)", re.I),
     "Dysuria"),

    # Defecation
    (re.compile(r"\b(?:it\s+)?(?:hurts?|burns?|stings?|bleeds?)\s+when\s+(?:i\s+)?(?:poop|defecate|go\s+to\s+the\s+toilet|have\s+a\s+bowel\s+movement)", re.I),
     "Painful defecation / rectal pain"),
    (re.compile(r"\bblood\s+(?:when|after)\s+(?:i\s+)?(?:wipe|poop|go\s+to\s+the\s+toilet)", re.I),
     "Hematochezia / rectal bleeding"),

    # Breathing
    (re.compile(r"\b(?:can't|cannot|hard to|trouble|difficulty)\s+(?:breathe|breathing|catch\s+(?:my\s+)?breath)", re.I),
     "Dyspnea"),
    (re.compile(r"\b(?:get|gets?|getting)\s+(?:out\s+of\s+breath|winded|breathless)\s+(?:easily|quickly|fast)", re.I),
     "Exertional dyspnea"),

    # Eating / swallowing
    (re.compile(r"\b(?:can't|cannot|hard to|trouble|difficulty)\s+(?:swallow|eat|keep\s+food\s+down)", re.I),
     "Dysphagia / nausea"),

    # Sleep
    (re.compile(r"\b(?:can't|cannot)\s+(?:sleep|fall\s+asleep|stay\s+asleep|get\s+to\s+sleep)", re.I),
     "Insomnia"),
    (re.compile(r"\b(?:wak(?:e|ing)\s+up\s+(?:in\s+the\s+night|at\s+night|throughout\s+the\s+night))", re.I),
     "Sleep disturbance / nocturia"),

    # Lump / mass
    (re.compile(r"\b(?:found|feel|noticed|there'?s?)\s+(?:a\s+)?(?:lump|bump|mass|nodule|growth)\s+(?:in|on|near|under)\s+(?:my\s+)?\w+", re.I),
     "Mass / lump — requires evaluation"),

    # Swollen glands
    (re.compile(r"\b(?:my\s+)?(?:glands?|lymph\s+nodes?)\s+(?:are\s+)?(?:swollen|up|enlarged)", re.I),
     "Lymphadenopathy"),
    (re.compile(r"\bswollen\s+glands?\b", re.I),
     "Lymphadenopathy"),
]

PAIN_CHARACTERS = [
    "sharp", "dull", "burning", "stabbing", "throbbing", "aching",
    "cramping", "shooting", "radiating", "pressure-like", "squeezing",
    "tearing", "gnawing", "constant", "intermittent", "colicky",
]

SEVERITY_TERMS = {
    # Mild
    "mild": "mild", "slight": "mild", "minor": "mild", "little": "mild",
    "a bit": "mild", "a little": "mild", "slightly": "mild", "not too bad": "mild",
    "manageable": "mild", "bearable": "mild",
    # Moderate
    "moderate": "moderate", "significant": "moderate", "noticeable": "moderate",
    "pretty bad": "moderate", "fairly bad": "moderate", "quite bad": "moderate",
    "bothering me": "moderate", "annoying": "moderate",
    # Severe
    "severe": "severe", "intense": "severe", "extreme": "severe",
    "excruciating": "severe", "unbearable": "severe", "terrible": "severe",
    "worst": "severe", "awful": "severe", "horrible": "severe", "agony": "severe",
    "really bad": "severe", "very bad": "severe", "super bad": "severe",
    "killing me": "severe", "kills me": "severe", "can't stand it": "severe",
    "can't take it": "severe", "so bad": "severe", "bad": "moderate",
}

NEGATION_TERMS = [
    r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenying\b", r"\bwithout\b",
    r"\bnegative for\b", r"\bno complaints of\b", r"\bno history of\b",
    r"\babsent\b", r"\bnever\b", r"\brules out\b", r"\brunning out\b",
]

TEMPORAL_PATTERNS = [
    (r"(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*(ago|prior)?", "onset"),
    (r"(since|for the (last|past))\s+(\d+\s*(day|days|week|weeks|month|months|year|years))", "duration"),
    (r"(started|began|onset)\s+(yesterday|today|last\s+\w+|\d+\s*\w+\s*ago)", "onset"),
    (r"(gradual|sudden|acute|chronic|intermittent|persistent|constant|occasional)", "character"),
]

FREQUENCY_TERMS = [
    "daily", "weekly", "monthly", "constantly", "occasionally", "frequently",
    "every day", "every week", "once a day", "twice a day", "three times",
    "at night", "in the morning", "episodic", "recurring",
]

ANATOMICAL_REGIONS = [
    "head", "neck", "chest", "abdomen", "back", "shoulder", "arm", "elbow",
    "wrist", "hand", "hip", "thigh", "leg", "knee", "ankle", "foot",
    "left", "right", "bilateral", "upper", "lower", "anterior", "posterior",
    "lumbar", "thoracic", "cervical", "sacral", "inguinal", "groin",
    "epigastric", "periumbilical", "right upper quadrant", "left upper quadrant",
    "right lower quadrant", "left lower quadrant",
]


# ─────────────────────────── Medication lexicon ─────────────────────────────

MEDICATION_TERMS = {
    # Analgesics
    "ibuprofen": "NSAID analgesic",
    "naproxen": "NSAID analgesic",
    "aspirin": "NSAID/antiplatelet",
    "acetaminophen": "Analgesic/antipyretic",
    "tylenol": "Analgesic/antipyretic (acetaminophen)",
    "advil": "NSAID analgesic (ibuprofen)",
    "motrin": "NSAID analgesic (ibuprofen)",
    "aleve": "NSAID analgesic (naproxen)",
    "tramadol": "Opioid analgesic",
    "oxycodone": "Opioid analgesic",
    "hydrocodone": "Opioid analgesic",
    "morphine": "Opioid analgesic",
    "codeine": "Opioid analgesic",
    "gabapentin": "Anticonvulsant / neuropathic pain",
    "pregabalin": "Anticonvulsant / neuropathic pain",

    # Antibiotics
    "amoxicillin": "Penicillin antibiotic",
    "augmentin": "Penicillin/β-lactamase inhibitor",
    "azithromycin": "Macrolide antibiotic",
    "zpack": "Macrolide antibiotic (azithromycin)",
    "z-pack": "Macrolide antibiotic (azithromycin)",
    "doxycycline": "Tetracycline antibiotic",
    "ciprofloxacin": "Fluoroquinolone antibiotic",
    "cipro": "Fluoroquinolone antibiotic (ciprofloxacin)",
    "levofloxacin": "Fluoroquinolone antibiotic",
    "metronidazole": "Antibiotic/antiprotozoal",
    "flagyl": "Antibiotic/antiprotozoal (metronidazole)",
    "cephalexin": "Cephalosporin antibiotic",
    "clindamycin": "Lincosamide antibiotic",
    "trimethoprim": "Antibiotic",
    "bactrim": "Sulfonamide antibiotic (trimethoprim-sulfamethoxazole)",
    "penicillin": "Penicillin antibiotic",
    "nitrofurantoin": "Antibiotic (UTI)",
    "macrobid": "Antibiotic (nitrofurantoin)",

    # Cardiovascular
    "lisinopril": "ACE inhibitor / antihypertensive",
    "enalapril": "ACE inhibitor",
    "amlodipine": "Calcium channel blocker",
    "norvasc": "Calcium channel blocker (amlodipine)",
    "metoprolol": "Beta-blocker",
    "atenolol": "Beta-blocker",
    "carvedilol": "Beta-blocker",
    "losartan": "ARB antihypertensive",
    "valsartan": "ARB antihypertensive",
    "hydrochlorothiazide": "Thiazide diuretic",
    "hctz": "Thiazide diuretic (hydrochlorothiazide)",
    "furosemide": "Loop diuretic",
    "lasix": "Loop diuretic (furosemide)",
    "spironolactone": "Potassium-sparing diuretic",
    "warfarin": "Anticoagulant (vitamin K antagonist)",
    "coumadin": "Anticoagulant (warfarin)",
    "apixaban": "Anticoagulant (DOAC)",
    "eliquis": "Anticoagulant (apixaban)",
    "rivaroxaban": "Anticoagulant (DOAC)",
    "xarelto": "Anticoagulant (rivaroxaban)",
    "clopidogrel": "Antiplatelet",
    "plavix": "Antiplatelet (clopidogrel)",
    "digoxin": "Cardiac glycoside",
    "amiodarone": "Antiarrhythmic",
    "nitroglycerin": "Nitrate vasodilator",

    # Statins / lipid
    "atorvastatin": "HMG-CoA reductase inhibitor (statin)",
    "lipitor": "Statin (atorvastatin)",
    "simvastatin": "Statin",
    "zocor": "Statin (simvastatin)",
    "rosuvastatin": "Statin",
    "crestor": "Statin (rosuvastatin)",

    # Diabetes
    "metformin": "Biguanide / antidiabetic",
    "glucophage": "Biguanide (metformin)",
    "insulin": "Insulin therapy",
    "glipizide": "Sulfonylurea antidiabetic",
    "glimepiride": "Sulfonylurea antidiabetic",
    "januvia": "DPP-4 inhibitor (sitagliptin)",
    "sitagliptin": "DPP-4 inhibitor",
    "ozempic": "GLP-1 receptor agonist (semaglutide)",
    "semaglutide": "GLP-1 receptor agonist",
    "jardiance": "SGLT-2 inhibitor (empagliflozin)",
    "empagliflozin": "SGLT-2 inhibitor",

    # Respiratory
    "albuterol": "Short-acting β2 agonist (SABA)",
    "ventolin": "SABA (albuterol)",
    "salbutamol": "SABA",
    "salmeterol": "Long-acting β2 agonist (LABA)",
    "fluticasone": "Inhaled corticosteroid",
    "advair": "ICS/LABA combination",
    "symbicort": "ICS/LABA combination",
    "montelukast": "Leukotriene receptor antagonist",
    "singulair": "Leukotriene antagonist (montelukast)",
    "tiotropium": "LAMA bronchodilator",
    "spiriva": "LAMA (tiotropium)",

    # GI
    "omeprazole": "Proton pump inhibitor",
    "prilosec": "PPI (omeprazole)",
    "pantoprazole": "Proton pump inhibitor",
    "protonix": "PPI (pantoprazole)",
    "esomeprazole": "Proton pump inhibitor",
    "nexium": "PPI (esomeprazole)",
    "ranitidine": "H2 blocker",
    "famotidine": "H2 blocker",
    "pepcid": "H2 blocker (famotidine)",
    "ondansetron": "Antiemetic (5-HT3 antagonist)",
    "zofran": "Antiemetic (ondansetron)",
    "metoclopramide": "Prokinetic antiemetic",
    "loperamide": "Antidiarrheal",
    "imodium": "Antidiarrheal (loperamide)",

    # Psych / neuro
    "sertraline": "SSRI antidepressant",
    "zoloft": "SSRI (sertraline)",
    "fluoxetine": "SSRI antidepressant",
    "prozac": "SSRI (fluoxetine)",
    "escitalopram": "SSRI antidepressant",
    "lexapro": "SSRI (escitalopram)",
    "citalopram": "SSRI antidepressant",
    "venlafaxine": "SNRI antidepressant",
    "effexor": "SNRI (venlafaxine)",
    "duloxetine": "SNRI antidepressant",
    "cymbalta": "SNRI (duloxetine)",
    "bupropion": "NDRI antidepressant / smoking cessation",
    "wellbutrin": "NDRI (bupropion)",
    "alprazolam": "Benzodiazepine anxiolytic",
    "xanax": "Benzodiazepine (alprazolam)",
    "lorazepam": "Benzodiazepine",
    "ativan": "Benzodiazepine (lorazepam)",
    "clonazepam": "Benzodiazepine",
    "klonopin": "Benzodiazepine (clonazepam)",
    "diazepam": "Benzodiazepine",
    "valium": "Benzodiazepine (diazepam)",
    "zolpidem": "Sedative-hypnotic",
    "ambien": "Sedative-hypnotic (zolpidem)",
    "quetiapine": "Atypical antipsychotic",
    "seroquel": "Atypical antipsychotic (quetiapine)",
    "olanzapine": "Atypical antipsychotic",
    "risperidone": "Atypical antipsychotic",
    "haloperidol": "Typical antipsychotic",
    "lithium": "Mood stabilizer",
    "valproate": "Anticonvulsant / mood stabilizer",
    "lamotrigine": "Anticonvulsant / mood stabilizer",
    "phenytoin": "Anticonvulsant",
    "levetiracetam": "Anticonvulsant",
    "topiramate": "Anticonvulsant",

    # Thyroid / hormones
    "levothyroxine": "Thyroid hormone replacement",
    "synthroid": "Thyroid hormone (levothyroxine)",
    "estradiol": "Estrogen therapy",
    "progesterone": "Progesterone therapy",
    "testosterone": "Testosterone therapy",
    "prednisone": "Systemic corticosteroid",
    "methylprednisolone": "Systemic corticosteroid",
    "hydrocortisone": "Corticosteroid",
    "dexamethasone": "Corticosteroid",

    # Allergy / immunology
    "cetirizine": "H1 antihistamine",
    "zyrtec": "H1 antihistamine (cetirizine)",
    "loratadine": "H1 antihistamine",
    "claritin": "H1 antihistamine (loratadine)",
    "diphenhydramine": "H1 antihistamine (sedating)",
    "benadryl": "H1 antihistamine (diphenhydramine)",
    "fexofenadine": "Non-sedating H1 antihistamine",
    "allegra": "H1 antihistamine (fexofenadine)",
    "epinephrine": "Sympathomimetic / anaphylaxis",
    "epipen": "Epinephrine auto-injector",

    # Supplements
    "vitamin d": "Vitamin D supplementation",
    "vitamin b12": "Vitamin B12 supplementation",
    "folic acid": "Folate supplementation",
    "iron": "Iron supplementation",
    "calcium": "Calcium supplementation",
    "magnesium": "Magnesium supplementation",
    "zinc": "Zinc supplementation",
    "omega-3": "Omega-3 fatty acid supplementation",
    "fish oil": "Omega-3 supplement",
    "multivitamin": "Multivitamin supplementation",
}

DOSE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(mg|mcg|g|ml|mL|units?|IU|meq|mEq|mmol|tablets?|tabs?|caps?|capsules?|puffs?|drops?)",
    re.IGNORECASE,
)
FREQUENCY_PATTERN = re.compile(
    r"\b(once|twice|three times|four times|every\s+\d+\s+hours?|"
    r"q\d+h|qd|bid|tid|qid|qhs|prn|daily|weekly|monthly|"
    r"every (morning|evening|night)|at bedtime|as needed)\b",
    re.IGNORECASE,
)
ROUTE_PATTERN = re.compile(
    r"\b(oral(?:ly)?|po|iv|intravenous(?:ly)?|im|intramuscular(?:ly)?|"
    r"subcutaneous(?:ly)?|sq|sc|topical(?:ly)?|inhaled?|sublingual(?:ly)?|"
    r"rectal(?:ly)?|transdermal(?:ly)?|intranasal(?:ly)?|ophthalmic(?:ally)?)\b",
    re.IGNORECASE,
)


# ─────────────────────────── Vitals extraction ──────────────────────────────

VITAL_PATTERNS = [
    # Blood pressure
    (re.compile(r"(?:bp|blood pressure)[:\s]+(\d{2,3})\s*/\s*(\d{2,3})", re.I),
     lambda m: Vital("Blood Pressure", f"{m.group(1)}/{m.group(2)}", "mmHg", m.group(0))),
    (re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?", re.I),
     lambda m: Vital("Blood Pressure", f"{m.group(1)}/{m.group(2)}", "mmHg", m.group(0))
     if 60 <= int(m.group(1)) <= 250 and 40 <= int(m.group(2)) <= 150 else None),

    # Heart rate / pulse
    (re.compile(r"(?:hr|heart rate|pulse)[:\s]+(\d{2,3})\s*(?:bpm)?", re.I),
     lambda m: Vital("Heart Rate", m.group(1), "bpm", m.group(0))),
    (re.compile(r"(\d{2,3})\s*bpm", re.I),
     lambda m: Vital("Heart Rate", m.group(1), "bpm", m.group(0))),

    # Temperature
    (re.compile(r"(?:temp(?:erature)?)[:\s]+([\d.]+)\s*(?:°?[FC])?", re.I),
     lambda m: Vital("Temperature", m.group(1), "°F", m.group(0))),
    (re.compile(r"(3[5-9]|4[0-2])\.\d\s*(?:°?[CF])", re.I),
     lambda m: Vital("Temperature", m.group(0).split()[0], "°C", m.group(0))),
    (re.compile(r"(9[5-9]|10[0-5])\.\d\s*(?:°?F)?", re.I),
     lambda m: Vital("Temperature", m.group(0).split()[0], "°F", m.group(0))),

    # O2 sat
    (re.compile(r"(?:o2 sat(?:uration)?|spo2|oxygen saturation|sats?)[:\s]+([\d.]+)\s*%?", re.I),
     lambda m: Vital("O2 Saturation", m.group(1), "%", m.group(0))),
    (re.compile(r"(9[0-9]|100)\s*%\s*(?:on\s+(?:room air|ra|O2))?", re.I),
     lambda m: Vital("O2 Saturation", m.group(1), "%", m.group(0))),

    # Respiratory rate
    (re.compile(r"(?:rr|resp(?:iratory)? rate)[:\s]+(\d{1,2})", re.I),
     lambda m: Vital("Respiratory Rate", m.group(1), "breaths/min", m.group(0))),

    # Weight
    (re.compile(r"(?:weight|wt)[:\s]+([\d.]+)\s*(kg|lbs?|pounds?)", re.I),
     lambda m: Vital("Weight", m.group(1), m.group(2), m.group(0))),
    (re.compile(r"([\d.]+)\s*(kg|lbs?)\b", re.I),
     lambda m: Vital("Weight", m.group(1), m.group(2), m.group(0))
     if 20 <= float(m.group(1)) <= 500 else None),

    # Height
    (re.compile(r"(?:height|ht)[:\s]+([\d.]+)\s*(cm|m\b|ft|feet|')", re.I),
     lambda m: Vital("Height", m.group(1), m.group(2), m.group(0))),
    (re.compile(r"(\d)\s*(?:feet|ft|')\s*(\d{1,2})\s*(?:inches?|in|\")?", re.I),
     lambda m: Vital("Height", f"{m.group(1)}'{m.group(2)}\"", "ft/in", m.group(0))),

    # BMI
    (re.compile(r"(?:bmi)[:\s]+([\d.]+)", re.I),
     lambda m: Vital("BMI", m.group(1), "kg/m²", m.group(0))),

    # Blood glucose
    (re.compile(r"(?:glucose|blood sugar|bs|bg|cbg|fbs)[:\s]+([\d.]+)\s*(?:mg/dl|mmol)?", re.I),
     lambda m: Vital("Blood Glucose", m.group(1), "mg/dL", m.group(0))),
]


# ─────────────────────────── ICD-10 mapping ─────────────────────────────────

ICD10_MAP = {
    # Respiratory
    "pneumonia": ("J18.9", "Pneumonia, unspecified"),
    "bronchitis": ("J40", "Bronchitis, not specified as acute or chronic"),
    "asthma": ("J45.909", "Unspecified asthma, uncomplicated"),
    "copd": ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"),
    "upper respiratory infection": ("J06.9", "Acute upper respiratory infection, unspecified"),
    "uri": ("J06.9", "Acute upper respiratory infection, unspecified"),
    "sinusitis": ("J32.9", "Chronic sinusitis, unspecified"),
    "pharyngitis": ("J02.9", "Acute pharyngitis, unspecified"),
    "strep throat": ("J02.0", "Streptococcal pharyngitis"),
    "influenza": ("J11.1", "Influenza with other respiratory manifestations"),
    "covid": ("U09.9", "Post-COVID-19 condition, unspecified"),

    # Cardiovascular
    "hypertension": ("I10", "Essential (primary) hypertension"),
    "high blood pressure": ("I10", "Essential (primary) hypertension"),
    "heart failure": ("I50.9", "Heart failure, unspecified"),
    "atrial fibrillation": ("I48.91", "Unspecified atrial fibrillation"),
    "afib": ("I48.91", "Unspecified atrial fibrillation"),
    "coronary artery disease": ("I25.10", "Atherosclerotic heart disease of native coronary artery"),
    "cad": ("I25.10", "Atherosclerotic heart disease of native coronary artery"),
    "myocardial infarction": ("I21.9", "Acute myocardial infarction, unspecified"),
    "heart attack": ("I21.9", "Acute myocardial infarction, unspecified"),
    "chest pain": ("R07.9", "Chest pain, unspecified"),
    "angina": ("I20.9", "Angina pectoris, unspecified"),
    "dvt": ("I82.409", "Deep vein thrombosis, unspecified"),
    "pulmonary embolism": ("I26.99", "Other pulmonary embolism without acute cor pulmonale"),

    # Endocrine / metabolic
    "diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "type 2 diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "type 1 diabetes": ("E10.9", "Type 1 diabetes mellitus without complications"),
    "hypothyroidism": ("E03.9", "Hypothyroidism, unspecified"),
    "hyperthyroidism": ("E05.90", "Thyrotoxicosis, unspecified, without thyrotoxic crisis"),
    "obesity": ("E66.9", "Obesity, unspecified"),
    "hyperlipidemia": ("E78.5", "Hyperlipidemia, unspecified"),
    "dyslipidemia": ("E78.5", "Hyperlipidemia, unspecified"),

    # GI
    "gerd": ("K21.0", "Gastroesophageal reflux disease with esophagitis"),
    "acid reflux": ("K21.9", "Gastroesophageal reflux disease without esophagitis"),
    "peptic ulcer": ("K27.9", "Peptic ulcer, unspecified"),
    "gastritis": ("K29.70", "Gastritis, unspecified, without bleeding"),
    "ibs": ("K58.9", "Irritable bowel syndrome without diarrhea"),
    "irritable bowel syndrome": ("K58.9", "Irritable bowel syndrome without diarrhea"),
    "colitis": ("K52.9", "Noninfective gastroenteritis and colitis, unspecified"),
    "appendicitis": ("K37", "Unspecified appendicitis"),
    "cholecystitis": ("K81.9", "Cholecystitis, unspecified"),

    # Musculoskeletal
    "osteoarthritis": ("M19.90", "Primary osteoarthritis, unspecified site"),
    "rheumatoid arthritis": ("M06.9", "Rheumatoid arthritis, unspecified"),
    "gout": ("M10.9", "Gout, unspecified"),
    "back pain": ("M54.5", "Low back pain"),
    "low back pain": ("M54.5", "Low back pain"),
    "cervicalgia": ("M54.2", "Cervicalgia"),
    "neck pain": ("M54.2", "Cervicalgia"),
    "fibromyalgia": ("M79.3", "Panniculitis"),
    "osteoporosis": ("M81.0", "Age-related osteoporosis without current pathological fracture"),
    "carpal tunnel": ("G56.00", "Carpal tunnel syndrome, unspecified upper limb"),

    # Neurological / psych
    "migraine": ("G43.909", "Migraine, unspecified, not intractable, without status migrainosus"),
    "headache": ("R51", "Headache"),
    "depression": ("F32.9", "Major depressive disorder, single episode, unspecified"),
    "anxiety": ("F41.9", "Anxiety disorder, unspecified"),
    "insomnia": ("G47.00", "Insomnia, unspecified"),
    "adhd": ("F90.9", "Attention-deficit hyperactivity disorder, unspecified type"),
    "bipolar": ("F31.9", "Bipolar disorder, unspecified"),
    "ptsd": ("F43.10", "Post-traumatic stress disorder, unspecified"),
    "epilepsy": ("G40.909", "Epilepsy, unspecified, not intractable, without status epilepticus"),
    "parkinson": ("G20", "Parkinson's disease"),
    "alzheimer": ("G30.9", "Alzheimer's disease, unspecified"),
    "dementia": ("F03.90", "Unspecified dementia without behavioral disturbance"),

    # Renal / urological
    "uti": ("N39.0", "Urinary tract infection, site not specified"),
    "urinary tract infection": ("N39.0", "Urinary tract infection, site not specified"),
    "kidney stones": ("N20.0", "Calculus of kidney"),
    "nephrolithiasis": ("N20.0", "Calculus of kidney"),
    "chronic kidney disease": ("N18.9", "Chronic kidney disease, unspecified"),
    "ckd": ("N18.9", "Chronic kidney disease, unspecified"),

    # Infectious
    "cellulitis": ("L03.90", "Cellulitis, unspecified"),
    "abscess": ("L02.91", "Cutaneous abscess, unspecified"),
    "sepsis": ("A41.9", "Sepsis, unspecified organism"),

    # Other common
    "anemia": ("D64.9", "Anemia, unspecified"),
    "allergic rhinitis": ("J30.9", "Allergic rhinitis, unspecified"),
    "eczema": ("L30.9", "Dermatitis, unspecified"),
    "psoriasis": ("L40.9", "Psoriasis, unspecified"),
    "sleep apnea": ("G47.33", "Obstructive sleep apnea (adult) (pediatric)"),
    "fatty liver": ("K76.0", "Fatty (change of) liver, not elsewhere classified"),
    "hypothyroidism": ("E03.9", "Hypothyroidism, unspecified"),
}

PLAN_KEYWORDS = {
    "prescribe": "prescribe",
    "prescribed": "prescribe",
    "start": "initiate",
    "begin": "initiate",
    "continue": "continue",
    "stop": "discontinue",
    "discontinue": "discontinue",
    "refer": "refer",
    "referral": "refer",
    "order": "order",
    "schedule": "schedule",
    "return": "follow-up",
    "follow up": "follow-up",
    "follow-up": "follow-up",
    "monitor": "monitor",
    "blood work": "laboratory",
    "lab": "laboratory",
    "labs": "laboratory",
    "x-ray": "imaging",
    "xray": "imaging",
    "mri": "imaging",
    "ct scan": "imaging",
    "ultrasound": "imaging",
    "ecg": "diagnostic",
    "ekg": "diagnostic",
    "echo": "diagnostic",
    "biopsy": "procedure",
    "injection": "procedure",
}

SOCIAL_PATTERNS = {
    "smoking": re.compile(r"\b(smok(?:es?|ing|er)|cigarette|tobacco|pack(?:\s+a\s+|\s+per\s+)(?:day|week))\b", re.I),
    "alcohol": re.compile(r"\b(alcohol|drink(?:ing|s)?|beer|wine|liquor|social drinker)\b", re.I),
    "drugs": re.compile(r"\b(drug use|illicit|recreational|marijuana|cannabis|cocaine|heroin|meth)\b", re.I),
    "exercise": re.compile(r"\b(exercis(?:e|ing|es)|workout|gym|walk(?:ing)?|run(?:ning)?|sedentary|active)\b", re.I),
    "occupation": re.compile(r"\b(works?\s+as|employed|occupation|retired|unemployed|disability)\b", re.I),
    "marital_status": re.compile(r"\b(married|single|divorced|widowed|partner)\b", re.I),
}

ALLERGY_PATTERNS = [
    re.compile(r"allerg(?:ic|y|ies)\s+to\s+([\w\s,]+?)(?:\.|,|;|\n|$)", re.I),
    re.compile(r"([\w\s]+?)\s+allergy", re.I),
    re.compile(r"nkda|no known drug allergies|no known allergies", re.I),
]

FAMILY_HISTORY_TERMS = [
    "family history", "mother", "father", "parents", "sibling", "brother", "sister",
    "grandfather", "grandmother", "grandparents", "familial", "hereditary", "runs in the family",
]


# ─────────────────────────── Main extraction class ──────────────────────────

class MedicalNLPEngine:

    def __init__(self):
        self._symptom_re = self._build_symptom_regex()
        self._med_re = self._build_medication_regex()

    def _build_symptom_regex(self) -> re.Pattern:
        terms = sorted(SYMPTOM_TERMS.keys(), key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def _build_medication_regex(self) -> re.Pattern:
        terms = sorted(MEDICATION_TERMS.keys(), key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def _is_negated(self, text: str, start: int, window: int = 60) -> bool:
        # Only look within the same sentence (don't cross . ? ! or newline)
        before_full = text[max(0, start - window):start]
        # Clip to last sentence boundary
        for sep in ['.', '!', '?', '\n']:
            idx = before_full.rfind(sep)
            if idx != -1:
                before_full = before_full[idx + 1:]
        before = before_full.lower()
        for neg in NEGATION_TERMS:
            if re.search(neg, before):
                return True
        return False

    def _get_context(self, text: str, start: int, end: int, window: int = 120) -> str:
        return text[max(0, start - window):min(len(text), end + window)]

    def _extract_severity(self, context: str) -> Optional[str]:
        for term, sev in SEVERITY_TERMS.items():
            if re.search(r"\b" + re.escape(term) + r"\b", context, re.I):
                return sev
        # Numeric scale
        m = re.search(r"(\d+)\s*(?:/|out of)\s*10", context, re.I)
        if m:
            n = int(m.group(1))
            if n <= 3: return "mild"
            if n <= 6: return "moderate"
            return "severe"
        return None

    def _extract_duration(self, context: str) -> Optional[str]:
        patterns = [
            r"for\s+(?:the\s+(?:past|last)\s+)?(\d+\s*(?:day|days|week|weeks|month|months|year|years))",
            r"since\s+(\w+\s*\w*)",
            r"(\d+\s*(?:day|days|week|weeks|month|months|year|years))\s*(?:ago|duration|history)",
            r"(yesterday|today|this morning|last night|last week|last month)",
        ]
        for pat in patterns:
            m = re.search(pat, context, re.I)
            if m:
                return m.group(1)
        return None

    def _extract_character(self, context: str) -> Optional[str]:
        found = [c for c in PAIN_CHARACTERS if re.search(r"\b" + re.escape(c) + r"\b", context, re.I)]
        return ", ".join(found) if found else None

    def _extract_location(self, context: str) -> Optional[str]:
        found = [r for r in ANATOMICAL_REGIONS if re.search(r"\b" + re.escape(r) + r"\b", context, re.I)]
        return ", ".join(found[:3]) if found else None

    def extract_symptoms(self, text: str) -> list[Symptom]:
        seen = set()
        symptoms = []

        # ── 1. Static keyword lexicon ────────────────────────────────────
        for m in self._symptom_re.finditer(text):
            raw = m.group(0).lower()
            canonical = SYMPTOM_TERMS[raw]
            if canonical in seen:
                continue
            seen.add(canonical)
            ctx = self._get_context(text, m.start(), m.end())
            symptoms.append(Symptom(
                name=canonical,
                negated=self._is_negated(text, m.start()),
                severity=self._extract_severity(ctx),
                duration=self._extract_duration(ctx),
                character=self._extract_character(ctx),
                location=self._extract_location(ctx),
                context=ctx.strip(),
            ))

        # ── 2. Dynamic colloquial phrase patterns ────────────────────────
        for pattern, canonical in COLLOQUIAL_BODY_PATTERNS:
            for m in pattern.finditer(text):
                if canonical in seen:
                    continue
                seen.add(canonical)
                ctx = self._get_context(text, m.start(), m.end())
                symptoms.append(Symptom(
                    name=canonical,
                    negated=self._is_negated(text, m.start()),
                    severity=self._extract_severity(ctx),
                    duration=self._extract_duration(ctx),
                    character=self._extract_character(ctx),
                    location=self._extract_location(ctx),
                    context=ctx.strip(),
                ))

        return symptoms

    def extract_medications(self, text: str) -> list[Medication]:
        seen = set()
        meds = []
        for m in self._med_re.finditer(text):
            raw = m.group(0).lower()
            if raw in seen:
                continue
            seen.add(raw)
            ctx = self._get_context(text, m.start(), m.end(), window=150)

            # Determine status
            status = "mentioned"
            ctx_lower = ctx.lower()
            if any(w in ctx_lower for w in ["prescribed", "started on", "start", "initiate", "begin"]):
                status = "prescribed"
            elif any(w in ctx_lower for w in ["currently taking", "currently on", "takes", "on "]):
                status = "current"
            elif any(w in ctx_lower for w in ["stopped", "discontinued", "no longer"]):
                status = "discontinued"
            elif any(w in ctx_lower for w in ["allergic to", "allergy to"]):
                status = "allergic"

            dose_m = DOSE_PATTERN.search(ctx)
            freq_m = FREQUENCY_PATTERN.search(ctx)
            route_m = ROUTE_PATTERN.search(ctx)

            meds.append(Medication(
                name=f"{m.group(0).title()} ({MEDICATION_TERMS[raw]})",
                dose=f"{dose_m.group(1)} {dose_m.group(2)}" if dose_m else None,
                frequency=freq_m.group(0) if freq_m else None,
                route=route_m.group(0) if route_m else None,
                status=status,
            ))
        return meds

    def extract_vitals(self, text: str) -> list[Vital]:
        vitals = []
        seen_names = set()
        for pattern, builder in VITAL_PATTERNS:
            for m in pattern.finditer(text):
                try:
                    v = builder(m)
                    if v is not None and v.name not in seen_names:
                        vitals.append(v)
                        seen_names.add(v.name)
                except (ValueError, AttributeError):
                    continue
        return vitals

    def extract_diagnoses(self, text: str) -> list[Diagnosis]:
        diagnoses = []
        seen = set()

        # Look for explicit diagnosis markers
        dx_patterns = [
            re.compile(r"(?:diagnos(?:ed?|is)|impression|assessment|working diagnosis)[:\s]+([^.\n]+)", re.I),
            re.compile(r"(?:consistent with|likely|probable|confirmed|ruling out)\s+([^.,\n]+)", re.I),
            re.compile(r"(?:history of|hx of|known)\s+([^.,\n]+)", re.I),
        ]

        for pat in dx_patterns:
            for m in pat.finditer(text):
                fragment = m.group(1).strip().lower()
                for condition, (code, desc) in ICD10_MAP.items():
                    if condition in fragment and condition not in seen:
                        seen.add(condition)
                        certainty = "confirmed"
                        if re.search(r"rule out|possible|probable|likely|r/o", m.group(0), re.I):
                            certainty = "possible"
                        if re.search(r"ruled out|not consistent", m.group(0), re.I):
                            certainty = "ruled-out"
                        diagnoses.append(Diagnosis(
                            name=desc,
                            icd10=code,
                            certainty=certainty,
                            primary=len(diagnoses) == 0,
                        ))

        # Fallback: scan full text for condition mentions
        for condition, (code, desc) in ICD10_MAP.items():
            if condition not in seen:
                if re.search(r"\b" + re.escape(condition) + r"\b", text, re.I):
                    seen.add(condition)
                    diagnoses.append(Diagnosis(
                        name=desc,
                        icd10=code,
                        certainty="possible",
                        primary=False,
                    ))

        return diagnoses[:10]  # Cap at 10

    def extract_allergies(self, text: str) -> list[str]:
        allergies = []
        for pat in ALLERGY_PATTERNS:
            for m in pat.finditer(text):
                raw = m.group(0).strip()
                if re.search(r"nkda|no known", raw, re.I):
                    return ["NKDA – No Known Drug Allergies"]
                if m.lastindex and m.group(1):
                    substances = [s.strip() for s in re.split(r",|and", m.group(1))]
                    allergies.extend(s.title() for s in substances if len(s) > 2)
        return list(set(allergies))[:10]

    def extract_social_history(self, text: str) -> dict:
        social = {}
        for category, pat in SOCIAL_PATTERNS.items():
            m = pat.search(text)
            if m:
                ctx = self._get_context(text, m.start(), m.end(), window=60)
                social[category] = ctx.strip()
        return social

    def extract_family_history(self, text: str) -> list[str]:
        items = []
        for term in FAMILY_HISTORY_TERMS:
            pattern = re.compile(r"(?:" + re.escape(term) + r")[^.]*\.", re.I)
            for m in pattern.finditer(text):
                items.append(m.group(0).strip())
                if len(items) >= 5:
                    return items
        return items

    def extract_plan_items(self, text: str) -> list[str]:
        items = []
        sentences = re.split(r"[.!?]\s+", text)
        for sent in sentences:
            sent_lower = sent.lower()
            for keyword, action in PLAN_KEYWORDS.items():
                if re.search(r"\b" + re.escape(keyword) + r"\b", sent_lower):
                    items.append(sent.strip())
                    break
        return list(dict.fromkeys(items))[:15]  # deduplicate, cap at 15

    def extract_review_of_systems(self, text: str) -> dict:
        systems = {
            "Constitutional": ["fever", "chills", "weight loss", "fatigue", "malaise", "night sweats"],
            "HEENT": ["headache", "blurred vision", "hearing loss", "tinnitus", "sore throat", "hoarseness"],
            "Respiratory": ["cough", "shortness of breath", "wheezing", "hemoptysis"],
            "Cardiovascular": ["chest pain", "palpitations", "edema", "leg swelling"],
            "Gastrointestinal": ["nausea", "vomiting", "diarrhea", "constipation", "heartburn", "abdominal pain"],
            "Genitourinary": ["frequent urination", "painful urination", "blood in urine", "incontinence"],
            "Musculoskeletal": ["joint pain", "back pain", "muscle pain", "stiffness"],
            "Neurological": ["dizziness", "numbness", "tingling", "weakness", "seizure", "confusion"],
            "Psychiatric": ["anxiety", "depression", "insomnia", "mood changes"],
            "Skin": ["rash", "itching", "bruising"],
            "Endocrine": ["excessive thirst", "excessive urination", "hot flashes", "cold intolerance"],
        }
        ros = {}
        for system, symptoms in systems.items():
            positive = []
            negative = []
            for s in symptoms:
                if re.search(r"\b" + re.escape(s) + r"\b", text, re.I):
                    if self._is_negated(text, text.lower().find(s)):
                        negative.append(s)
                    else:
                        positive.append(s)
            if positive or negative:
                ros[system] = {"positive": positive, "negative": negative}
        return ros

    def analyze(self, text: str) -> ClinicalEntities:
        return ClinicalEntities(
            symptoms=self.extract_symptoms(text),
            medications=self.extract_medications(text),
            vitals=self.extract_vitals(text),
            diagnoses=self.extract_diagnoses(text),
            allergies=self.extract_allergies(text),
            procedures=[],
            family_history=self.extract_family_history(text),
            social_history=self.extract_social_history(text),
            review_of_systems=self.extract_review_of_systems(text),
            plan_items=self.extract_plan_items(text),
        )
