"""
nlp_engine/relations.py
Relation extraction between clinical entities.
Detects: symptom→medication, symptom→dose, medication→indication, etc.
"""

import re
from typing import List, Optional, Tuple


# Causal / treatment relations
_TREATING_RE = re.compile(
    r"(\w[\w\s]+?)\s+(?:for|treating|to treat|because of|due to)\s+([\w\s,]+?)(?:[.;]|$)",
    re.I,
)

_CAUSED_BY_RE = re.compile(
    r"([\w\s]+?)\s+(?:caused by|secondary to|due to|as a result of)\s+([\w\s,]+?)(?:[.;]|$)",
    re.I,
)

# Drug–allergy relation
_ALLERGY_RE = re.compile(
    r"(?:allergic to|allergy to|reaction to|cannot take|can't take)\s+([\w\s,]+?)(?:[.;,]|$)",
    re.I,
)

# Family history relation
_FAMILY_RE = re.compile(
    r"(?:mother|father|parent|sibling|brother|sister|son|daughter|grandparent|family)\s+"
    r"(?:had|has|died of|diagnosed with|history of)\s+([\w\s,]+?)(?:[.;]|$)",
    re.I,
)

# Social history fragments
_SMOKING_RE = re.compile(
    r"\b(smok(?:es?|ed?|ing)|(?:quit|stopped)\s+smoking|(?:\d+)\s+pack[\s-]year|non[\s-]?smoker|never\s+smoked)\b",
    re.I,
)
_ALCOHOL_RE = re.compile(
    r"\b(drink(?:s|ing)?|alcohol|(?:\d+)\s+drink(?:s)?\s+(?:per|a)\s+(?:week|day|month)|social drinker|non[\s-]?drinker|sober)\b",
    re.I,
)
_DRUG_RE = re.compile(
    r"\b(recreational drug|drug use|cocaine|heroin|marijuana|cannabis|illicit|IV drug|intravenous drug)\b",
    re.I,
)
_EXERCISE_RE = re.compile(
    r"\b(exercises?|work(?:s)?\s+out|physically\s+active|sedentary|gym|jog(?:s|ging)?|walk(?:s|ing)?\s+\d+)\b",
    re.I,
)
_OCCUPATION_RE = re.compile(
    r"\b(?:works?\s+as|job\s+is|employed\s+as|occupation[:\s]+)([\w\s]+?)(?:[.;,]|$)",
    re.I,
)


def extract_allergies(text: str) -> List[str]:
    allergies = []
    for m in _ALLERGY_RE.finditer(text):
        raw = m.group(1).strip()
        for item in re.split(r",|and ", raw):
            item = item.strip()
            if 2 < len(item) < 60:
                allergies.append(item.title())
    return list(dict.fromkeys(allergies))


def extract_family_history(text: str) -> List[str]:
    items = []
    for m in _FAMILY_RE.finditer(text):
        items.append(m.group(0).strip()[:120])
    return items


def extract_social_history(text: str) -> dict:
    social = {}

    m = _SMOKING_RE.search(text)
    if m:
        social["smoking"] = m.group(0).strip()

    m = _ALCOHOL_RE.search(text)
    if m:
        social["alcohol"] = m.group(0).strip()

    m = _DRUG_RE.search(text)
    if m:
        social["recreational_drugs"] = m.group(0).strip()

    m = _EXERCISE_RE.search(text)
    if m:
        social["exercise"] = m.group(0).strip()

    m = _OCCUPATION_RE.search(text)
    if m:
        social["occupation"] = m.group(1).strip()

    return social


def extract_treatment_relations(text: str) -> List[Tuple[str, str]]:
    """Returns list of (treatment, condition) tuples."""
    relations = []
    for m in _TREATING_RE.finditer(text):
        relations.append((m.group(1).strip(), m.group(2).strip()))
    return relations
