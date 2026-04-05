// TypeScript types mirroring backend schemas (AGENTS.md v2)

export interface ClinicalEntity {
  text: string;
  type: string;
  start: number;
  end: number;
  negated: boolean;
  uncertain: boolean;
  confidence: number;
}

export interface DialogueActType {
  sentence: string;
  sentence_index: number;
  label: string;
  confidence: number;
  speaker: string;
}

export interface TemporalEventType {
  text: string;
  type: string;
  normalized: string;
  start: number;
  end: number;
}

export interface ClinicalEntities {
  symptoms: ClinicalEntity[];
  medications: ClinicalEntity[];
  diagnoses: ClinicalEntity[];
  vitals: ClinicalEntity[];
  dialogue_acts: DialogueActType[];
  temporal_events: TemporalEventType[];
  sentences: string[];
  negation_scopes: Array<{ text: string; type: string; start: number; end: number }>;
}

export interface Differential {
  diagnosis: string;
  evidence: string;
  likelihood: string;
  kb_source: string;
}

export interface SOAPNote {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  differentials: Differential[];
}

export interface SentenceVerification {
  soap_sentence: string;
  soap_section: string;
  label: string;
  confidence: number;
  source_transcript_sentence: string;
  is_hallucinated: boolean;
}

export interface VerificationResult {
  sentence_results: SentenceVerification[];
  faithfulness_score: number;
  hallucinated_sentences: SentenceVerification[];
}

export interface GCISResponse {
  transcript: string;
  patient_info: { patient_name?: string; patient_age?: number; patient_id?: string };
  entities: ClinicalEntities;
  soap: SOAPNote;
  verification: VerificationResult;
  refinement_iterations: number;
  pipeline_version: string;
}

export interface PatientInfo {
  patient_name: string;
  patient_age: number | string;
  patient_id: string;
}
