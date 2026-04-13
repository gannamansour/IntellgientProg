"""
rule_based_system/expert_system.py
Rule-based heart disease risk assessment using Experta.
10 rules based on common clinical guidelines.
"""

from experta import *


class PatientData(Fact):
    """Holds a single patient's raw (un-normalised) values."""
    age       = Field(float, mandatory=True)
    chol      = Field(float, mandatory=True)   # mg/dL
    trestbps  = Field(float, mandatory=True)   # resting BP mmHg
    thalach   = Field(float, mandatory=True)   # max heart rate
    oldpeak   = Field(float, mandatory=True)   # ST depression
    cp        = Field(int,   mandatory=True)   # chest-pain type 0-3
    exang     = Field(int,   mandatory=True)   # exercise angina 0/1
    fbs       = Field(int,   mandatory=True)   # fasting blood sugar >120 mg/dL 0/1
    ca        = Field(int,   mandatory=True)   # number of major vessels 0-3
    thal      = Field(int,   mandatory=True)   # thal: 1=normal,2=fixed defect,3=reversable


class HeartRiskEngine(KnowledgeEngine):
    """Inference engine with 10 clinical rules."""

    def __init__(self):
        super().__init__()
        self.risk_score = 0   # counts how many "high-risk" rules fire
        self.fired      = []  # human-readable explanations

    # ── Rule 1 ──────────────────────────────────────────────────────────
    @Rule(PatientData(chol=P(lambda c: c > 240), age=P(lambda a: a > 50)))
    def high_cholesterol_elder(self):
        self.risk_score += 1
        self.fired.append("High cholesterol (>240) AND age >50 → elevated risk")

    # ── Rule 2 ──────────────────────────────────────────────────────────
    @Rule(PatientData(trestbps=P(lambda bp: bp > 140)))
    def high_blood_pressure(self):
        self.risk_score += 1
        self.fired.append("Resting BP >140 mmHg → elevated risk")

    # ── Rule 3 ──────────────────────────────────────────────────────────
    @Rule(PatientData(cp=P(lambda cp: cp in (1, 2, 3))))
    def chest_pain_present(self):
        self.risk_score += 1
        self.fired.append("Chest pain reported → elevated risk")

    # ── Rule 4 ──────────────────────────────────────────────────────────
    @Rule(PatientData(exang=1))
    def exercise_induced_angina(self):
        self.risk_score += 1
        self.fired.append("Exercise-induced angina → elevated risk")

    # ── Rule 5 ──────────────────────────────────────────────────────────
    @Rule(PatientData(oldpeak=P(lambda op: op > 2.0)))
    def high_st_depression(self):
        self.risk_score += 1
        self.fired.append("ST depression >2.0 → elevated risk")

    # ── Rule 6 ──────────────────────────────────────────────────────────
    @Rule(PatientData(fbs=1))
    def high_fasting_sugar(self):
        self.risk_score += 1
        self.fired.append("Fasting blood sugar >120 mg/dL → elevated risk")

    # ── Rule 7 ──────────────────────────────────────────────────────────
    @Rule(PatientData(ca=P(lambda ca: ca >= 2)))
    def blocked_vessels(self):
        self.risk_score += 1
        self.fired.append("2+ major vessels blocked → elevated risk")

    # ── Rule 8 ──────────────────────────────────────────────────────────
    @Rule(PatientData(thal=P(lambda t: t in (2, 3))))
    def thalassemia_defect(self):
        self.risk_score += 1
        self.fired.append("Thalassemia defect detected → elevated risk")

    # ── Rule 9 ──────────────────────────────────────────────────────────
    @Rule(PatientData(thalach=P(lambda hr: hr < 100)))
    def low_max_heart_rate(self):
        self.risk_score += 1
        self.fired.append("Max heart rate <100 bpm → elevated risk")

    # ── Rule 10 ─────────────────────────────────────────────────────────
    @Rule(PatientData(age=P(lambda a: a > 60),
                      chol=P(lambda c: c > 200),
                      trestbps=P(lambda bp: bp > 130)))
    def elderly_multi_risk(self):
        self.risk_score += 1
        self.fired.append("Age >60 + cholesterol >200 + BP >130 → high compound risk")


def assess_patient(patient: dict) -> dict:
    """
    Run the rule engine on one patient dict and return a risk verdict.

    Expected keys (raw / un-normalised values):
        age, chol, trestbps, thalach, oldpeak, cp, exang, fbs, ca, thal
    """
    engine = HeartRiskEngine()
    engine.reset()
    engine.declare(PatientData(**{k: float(v) if k not in
                                  ("cp","exang","fbs","ca","thal") else int(v)
                                  for k, v in patient.items()}))
    engine.run()

    if engine.risk_score == 0:
        verdict = "LOW RISK"
    elif engine.risk_score <= 3:
        verdict = "MODERATE RISK"
    else:
        verdict = "HIGH RISK"

    return {
        "risk_score": engine.risk_score,
        "verdict":    verdict,
        "rules_fired": engine.fired,
    }


# ── Quick demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = dict(age=62, chol=260, trestbps=145, thalach=95,
                  oldpeak=2.5, cp=2, exang=1, fbs=1, ca=2, thal=3)
    result = assess_patient(sample)
    print(f"\nVerdict : {result['verdict']}  (score={result['risk_score']})")
    print("Rules fired:")
    for r in result["rules_fired"]:
        print("  •", r)
