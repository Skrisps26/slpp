"""
Quick test suite for GCIS backend. No GPU needed for most tests.
"""
import sys
sys.path.insert(0, 'backend')

def test_schemas():
    from schemas.entities import ClinicalEntity, ClinicalEntities
    from schemas.soap import SOAPNote, Differential
    from schemas.verification import VerificationResult
    from schemas.response import GCISResponse

    e_pos = ClinicalEntity(text='fever', type='SYMPTOM', start=7, end=12)
    e_neg = ClinicalEntity(text='cough', type='SYMPTOM', start=20, end=25, negated=True)
    entities = ClinicalEntities(
        symptoms=[e_pos, e_neg], medications=[], diagnoses=[], vitals=[],
        dialogue_acts=[], temporal_events=[], sentences=['Fever. Cough.'], negation_scopes=[])
    assert len(entities.confirmed_symptoms()) == 1
    assert len(entities.denied_symptoms()) == 1
    assert entities.confirmed_symptoms()[0].text == 'fever'

    diff = Differential(diagnosis='URI', evidence='fever', likelihood='moderate', kb_source='')
    note = SOAPNote(subjective='Pt has fever.', objective='T=101.2', assessment='URI', plan='rest', differentials=[diff])
    assert note.to_dict()['differentials'][0]['diagnosis'] == 'URI'

    resp = GCISResponse(transcript='test', patient_info={}, entities={}, soap={}, verification={}, refinement_iterations=0)
    assert resp.to_dict()['pipeline_version'] == '2.0.0'
    print('  PASS schemas')

def test_dialogue_act():
    from models.dialogue_act import DialogueActModel
    model = DialogueActModel.load()
    tests = [
        ('I have a sharp pain in my chest.', 'SYMPTOM_REPORT'),
        ('Do I need to take this with food?', 'QUESTION'),
        ('Your blood work confirms type 2 diabetes.', 'DIAGNOSIS_STATEMENT'),
        ('Take ibuprofen 500mg twice daily.', 'TREATMENT_PLAN'),
        ('Your test results look completely normal.', 'REASSURANCE'),
        ('I have been on lisinopril for five years.', 'HISTORY'),
        ('Please sign the consent form.', 'OTHER'),
    ]
    correct = 0
    for sentence, expected in tests:
        result = model.classify(sentence)
        if result['label'] == expected: correct += 1
        status = 'PASS' if result['label'] == expected else 'FAIL'
        print(f'    {status} expected={expected:25s} got={result["label"]:25s} "{sentence[:50]}"')
    assert correct == len(tests), f'Only {correct}/{len(tests)} dialogue act predictions correct!'
    print(f'  PASS dialogue_act ({correct}/{len(tests)})')

def test_ner_regex():
    from models.clinical_ner import ClinicalNERModel
    # When no fine-tuned model, should fall back to regex
    ner = ClinicalNERModel()
    text = 'Patient reports fever and cough. Taking ibuprofen for pain. No history of diabetes. Temperature is 101.2 F.'
    entities = ner.extract_entities(text)
    types = set(e.entity_type for e in entities)
    texts = [e.text for e in entities]
    print(f'    Found: {entities}')
    assert len(entities) > 0, 'NER should return at least some regex matches'
    print(f'  PASS ner_regex ({len(entities)} entities)')

def test_transcriber():
    from pipeline.transcriber import Transcriber
    t = Transcriber()
    assert hasattr(t, 'transcribe_bytes')
    assert hasattr(t, 'transcriber') or hasattr(t, 'transcribe_file')
    import uuid
    # Check that transcribe_bytes uses uuid
    import inspect
    src = inspect.getsource(t.transcribe_bytes)
    assert 'uuid' in src, 'transcribe_bytes should use uuid for temp files'
    print('  PASS transcriber (uuid tempfile check)')

def test_imports():
    from pipeline.orchestrator import GCISOrchestrator
    from pipeline.refiner import RefinementLayer
    from pipeline.pdf_exporter import PDFExporter
    from rag.retriever import KBDocument
    print('  PASS imports')

def test_ollama_model():
    # Check that generator and refiner use the correct model
    import pipeline.generator as gen
    import pipeline.refiner as ref
    assert 'qwen2.5:3b-instruct' in gen.MODEL, f'Generator uses wrong model: {gen.MODEL}'
    assert 'qwen2.5:3b-instruct' in ref.MODEL, f'Refiner uses wrong model: {ref.MODEL}'
    print(f'  PASS model={gen.MODEL}')

if __name__ == '__main__':
    print('\n=== GCIS Backend Test Suite ===\n')
    test_schemas()
    test_dialogue_act()
    test_ner_regex()
    test_transcriber()
    test_imports()
    test_ollama_model()
    print('\n=== ALL TESTS PASSED ===\n')
