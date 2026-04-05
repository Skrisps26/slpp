// TypeScript types mirroring backend schemas

export interface Entity {
  text: string;
  entity_type: string;
  start: number;
  end: number;
  confidence: number;
  negated: boolean;
}

export interface TemporalEvent {
  text: string;
  temporal_type: string;
  normalized: string;
  start: number;
  end: number;
}

export interface DialogueAct {
  text: string;
  label: string;
  confidence: number;
}

export interface ClinicalEntities {
  symptoms: Entity[];
  medications: Entity[];
  diagnoses: Entity[];
  vitals: Entity[];
  negation_scopes: Array<{
    text: string;
    type: string;
    start: number;
    end: number;
  }>;
  dialogue_acts: DialogueAct[];
  temporal_events: TemporalEvent[];
  sentences: string[];
}

export interface DifferentialDiagnosis {
  diagnosis: string;
  evidence: string;
  likelihood: string;
}

export interface SOAPNote {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  differentials: DifferentialDiagnosis[];
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
  entities: ClinicalEntities;
  soap: SOAPNote;
  verification: VerificationResult;
  refinement_iterations: number;
}

export interface PatientInfo {
  patient_name: string;
  patient_age: number;
  patient_id: string;
}
