"""
MedScribe – AI-Powered Clinical Documentation System
Records doctor-patient interactions and generates comprehensive SOAP reports.
All processing is fully local. No API keys or internet required.
"""

import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import threading
from datetime import datetime
from pathlib import Path

from recorder import AudioRecorder
from transcriber import transcribe, load_transcript_from_file
from nlp_engine import MedicalNLPEngine
from soap_builder import SOAPBuilder
from pdf_builder import PDFBuilder

BANNER = r"""
  __  __          _ ____            _ _
 |  \/  | ___  __| / ___|  ___ _ __(_) |__   ___
 | |\/| |/ _ \/ _` \___ \ / __| '__| | '_ \ / _ \
 | |  | |  __/ (_| |___) | (__| |  | | |_) |  __/
 |_|  |_|\___|\__,_|____/ \___|_|  |_|_.__/ \___|
"""

# ─── Optional Rich for fancy output ─────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich import box
    console = Console()
    RICH = True
except ImportError:
    console = None
    RICH = False


def cprint(msg: str, style: str = ""):
    if RICH:
        console.print(msg, style=style)
    else:
        import re
        print(re.sub(r"\[.*?\]", "", msg))


def cinput(prompt: str, default: str = "") -> str:
    if RICH:
        return Prompt.ask(prompt, default=default)
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def cconfirm(prompt: str, default: bool = False) -> bool:
    if RICH:
        return Confirm.ask(prompt, default=default)
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    return (raw.startswith("y")) if raw else default


# ─── Banner ──────────────────────────────────────────────────────────────────

def display_banner():
    if RICH:
        console.print(f"[bold cyan]{BANNER}[/bold cyan]")
        console.print(Panel(
            "[bold white]Record · Transcribe · Extract · Document[/bold white]\n"
            "[dim]Fully Local Medical Documentation · No API Required[/dim]",
            border_style="cyan", padding=(0, 4),
        ))
    else:
        print(BANNER)
        print("=" * 55)
        print("  MedScribe – Fully Local Clinical Documentation")
        print("=" * 55)
    print()


# ─── Patient info collection ─────────────────────────────────────────────────

def collect_patient_info() -> dict:
    cprint("[bold yellow]━━━  Patient & Encounter Information  ━━━[/bold yellow]")
    cprint("[dim]Press Enter to accept defaults.[/dim]\n")

    info = {}
    info["patient_name"]     = cinput("[bold]Patient Name[/bold]", "Anonymous")
    info["patient_dob"]      = cinput("Date of Birth (YYYY-MM-DD)", "Unknown")
    info["patient_id"]       = cinput("Patient ID / MRN", f"PT-{datetime.now().strftime('%Y%m%d%H%M')}")
    print()
    info["doctor_name"]      = cinput("[bold]Physician Name[/bold]", "Dr. Unknown")
    info["doctor_specialty"] = cinput("Specialty", "General Practice")
    info["facility"]         = cinput("Facility / Clinic", "Medical Center")

    if RICH:
        info["encounter_type"] = Prompt.ask(
            "Encounter Type",
            choices=["Office Visit", "Follow-up", "Urgent Care", "Telehealth", "Consultation", "Emergency"],
            default="Office Visit",
        )
    else:
        info["encounter_type"] = cinput(
            "Encounter Type (Office Visit/Follow-up/Urgent Care/Telehealth/Consultation/Emergency)",
            "Office Visit",
        )

    print()
    if cconfirm("[dim]Add pre-existing clinical context before recording?[/dim]", default=False):
        info["chief_complaint"]     = cinput("Chief Complaint", "")
        info["known_conditions"]    = cinput("Known Conditions (comma-separated)", "")
        info["current_medications"] = cinput("Current Medications (comma-separated)", "")
        info["allergies"]           = cinput("Allergies", "NKDA")
    else:
        info["chief_complaint"]     = ""
        info["known_conditions"]    = ""
        info["current_medications"] = ""
        info["allergies"]           = ""

    info["encounter_datetime"] = datetime.now().isoformat()
    return info


# ─── Recording mode ──────────────────────────────────────────────────────────

def recording_mode(patient_info: dict):
    recorder = AudioRecorder(output_dir="recordings")
    cprint("\n[bold green]━━━  Recording Session  ━━━[/bold green]")
    cprint("Press [bold]ENTER[/bold] to begin recording.\n")
    input()

    audio_path = recorder.start()
    cprint(f"[bold red]● RECORDING[/bold red]  [dim]Press ENTER to stop.[/dim]  File: {audio_path}")

    _stop = threading.Event()
    def _timer():
        t0 = time.time()
        while not _stop.is_set():
            e = int(time.time() - t0)
            sys.stdout.write(f"\r  {e//60:02d}:{e%60:02d} elapsed  ")
            sys.stdout.flush()
            time.sleep(1)
    threading.Thread(target=_timer, daemon=True).start()

    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        audio_path = recorder.stop()

    print()
    dur = recorder.get_duration()
    cprint(f"[green]✓  Recording saved[/green]  ({dur:.1f}s)  →  {audio_path}")

    cprint("\n[cyan]Transcribing audio...[/cyan]")
    transcript, method = transcribe(audio_path, progress_callback=lambda m: cprint(f"  [dim]{m}[/dim]"))
    cprint(f"[green]✓  Transcription complete[/green]  [dim](method: {method})[/dim]")
    return transcript, audio_path


# ─── Text / file mode ────────────────────────────────────────────────────────

def text_mode(patient_info: dict, text_file: str = None):
    if text_file:
        cprint(f"\n[cyan]Loading transcript from file:[/cyan] {text_file}")
        transcript = load_transcript_from_file(text_file)
        cprint(f"[green]✓  Loaded {len(transcript)} characters[/green]")
    else:
        cprint("\n[bold yellow]━━━  Manual Transcript Entry  ━━━[/bold yellow]")
        cprint("[dim]Paste transcript below. Press Enter twice on a blank line to finish.[/dim]\n")
        lines = []
        blank = 0
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                blank += 1
                if blank >= 2:
                    break
                lines.append(line)
            else:
                blank = 0
                lines.append(line)
        transcript = "\n".join(lines).strip() or "[No transcript provided.]"
    return transcript, None


# ─── Report generation ───────────────────────────────────────────────────────

def generate_report(transcript: str, patient_info: dict):
    cprint("\n[cyan]Running medical NLP pipeline...[/cyan]")

    pre_existing = {
        "current_medications": patient_info.get("current_medications", ""),
        "known_conditions":    patient_info.get("known_conditions", ""),
        "allergies_list":      [a.strip() for a in patient_info.get("allergies", "NKDA").split(",") if a.strip()],
    }

    engine   = MedicalNLPEngine()
    entities = engine.analyze(transcript)

    cprint(f"  [green]✓[/green] Symptoms:    {len(entities.symptoms)}")
    cprint(f"  [green]✓[/green] Medications: {len(entities.medications)}")
    cprint(f"  [green]✓[/green] Vitals:      {len(entities.vitals)}")
    cprint(f"  [green]✓[/green] Diagnoses:   {len(entities.diagnoses)}")
    cprint(f"  [green]✓[/green] Plan items:  {len(entities.plan_items)}")

    cprint("\n[cyan]Building SOAP note...[/cyan]")
    soap_note = SOAPBuilder().build(entities, transcript, patient_info, pre_existing)
    cprint("  [green]✓[/green] SOAP note assembled")

    if soap_note.clinical_flags:
        cprint(f"\n  [bold red]⚠  {len(soap_note.clinical_flags)} clinical alert(s) detected[/bold red]")
        for flag in soap_note.clinical_flags[:4]:
            cprint(f"    [red]{flag}[/red]")

    cprint("\n[cyan]Generating PDF...[/cyan]")
    pdf_path = PDFBuilder(output_dir="reports").build(soap_note)
    cprint(f"  [bold green]✓  Report saved:[/bold green] {pdf_path}")
    return pdf_path, soap_note


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(note, pdf_path: str):
    if RICH:
        t = Table(title="Encounter Summary", box=box.ROUNDED, border_style="cyan")
        t.add_column("Field", style="bold cyan", min_width=20)
        t.add_column("Value", style="white")
        t.add_row("Patient",          note.patient_name)
        t.add_row("Date",             note.encounter_date)
        t.add_row("Physician",        note.physician_name)
        t.add_row("Chief Complaint",  note.chief_complaint[:70])
        if note.primary_diagnosis:
            t.add_row("Primary Dx",   note.primary_diagnosis.name)
            t.add_row("ICD-10",       note.primary_diagnosis.icd10 or "–")
        t.add_row("Active Symptoms",  str(len([s for s in note.entities.symptoms if not s.negated])))
        t.add_row("Medications",      str(len(note.medications_current) + len(note.medications_prescribed)))
        t.add_row("Clinical Alerts",  str(len(note.clinical_flags)))
        t.add_row("PDF Report",       pdf_path)
        console.print()
        console.print(t)
    else:
        print("\n━━━ Summary ━━━")
        print(f"  Patient:   {note.patient_name}")
        print(f"  Date:      {note.encounter_date}")
        print(f"  Report:    {pdf_path}")


# ─── Demo transcript ─────────────────────────────────────────────────────────

DEMO_TRANSCRIPT = """
Doctor: Good morning, Ms. Smith. What brings you in today?

Patient: Hi doctor. I've been having really bad headaches for the past two weeks.
They're mostly on the right side, throbbing, and they get worse with bright light.
I'd say the pain is about a seven out of ten.

Doctor: Any nausea or vomiting?

Patient: Yes, nausea with a few of them. No vomiting though.

Doctor: Any vision changes, weakness, or numbness?

Patient: No numbness or weakness. My vision gets a bit blurry during the worst ones.

Doctor: You're on lisinopril for hypertension, correct?

Patient: Yes, lisinopril 10mg daily. I checked at the pharmacy last week —
my blood pressure was 158 over 92.

Doctor: Let me get your vitals. Blood pressure today is 162 over 94,
heart rate 88, temperature 98.6, oxygen saturation 97%.

Patient: Is that concerning?

Doctor: A bit high. How's your diabetes? Still taking metformin?

Patient: Metformin 500mg twice daily. Morning blood sugar has been around
180 to 210. I haven't been great with my diet.

Doctor: Any fatigue, increased thirst, or frequent urination?

Patient: Definitely more tired. More frequent urination — three or four times at night.

Doctor: Any fever, chills, recent infections?

Patient: No fever or chills. I had a cold three weeks ago but it cleared up.

Doctor: Family history of migraines?

Patient: Yes, my mother had migraines her whole life.

Doctor: Any medication allergies?

Patient: Penicillin — I get a rash.

Doctor: Do you smoke or drink?

Patient: I quit smoking five years ago. I drink socially, one glass of wine on weekends.

Doctor: Based on your history, symptoms, and family history, this presentation is
consistent with migraines. Your hypertension is also not optimally controlled,
and your blood glucose suggests your diabetes management needs adjustment.

I'm prescribing sumatriptan 50mg for acute migraine attacks.
I'll increase lisinopril to 20mg daily. Continue metformin.

I want labs — HbA1c, comprehensive metabolic panel, and lipid panel.
Monitor blood pressure twice daily at home. If it exceeds 180 over 110,
go to the emergency room immediately.

Patient: Okay, I understand.

Doctor: Follow up in four weeks to review your labs and headache response.
If headaches worsen or you develop sudden severe headache, vision loss,
or any new neurological symptoms, come in immediately or go to the ER.

Patient: Thank you, doctor.
"""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="medscribe",
        description="MedScribe – Fully Local Clinical Documentation System",
    )
    parser.add_argument("--mode", choices=["record", "text", "file"],
                        default="record",
                        help="Input mode: record (mic), text (paste), file (txt)")
    parser.add_argument("--transcript", type=str, default=None,
                        help="Path to .txt transcript file (use with --mode file)")
    parser.add_argument("--demo", action="store_true",
                        help="Run with built-in demo transcript")
    args = parser.parse_args()

    display_banner()

    if args.demo:
        cprint("[bold magenta]━━━  DEMO MODE  ━━━[/bold magenta]\n")
        patient_info = {
            "patient_name":        "Jane Smith",
            "patient_dob":         "1978-04-22",
            "patient_id":          "MRN-78042201",
            "doctor_name":         "Dr. Michael Chen",
            "doctor_specialty":    "Internal Medicine",
            "facility":            "Riverside Medical Center",
            "encounter_type":      "Office Visit",
            "chief_complaint":     "",
            "known_conditions":    "Hypertension, Type 2 Diabetes",
            "current_medications": "Metformin 500mg, Lisinopril 10mg",
            "allergies":           "Penicillin",
        }
        transcript = DEMO_TRANSCRIPT
    else:
        patient_info = collect_patient_info()
        if args.mode == "record":
            transcript, _ = recording_mode(patient_info)
        elif args.mode == "file":
            transcript, _ = text_mode(patient_info, text_file=args.transcript)
        else:
            transcript, _ = text_mode(patient_info, text_file=None)

    pdf_path, soap_note = generate_report(transcript, patient_info)
    print_summary(soap_note, pdf_path)
    cprint("\n[bold green]✓  Done![/bold green]  Open the PDF to review the clinical documentation.\n")


if __name__ == "__main__":
    main()
