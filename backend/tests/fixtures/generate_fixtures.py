"""Generate synthetic test fixtures with realistic business characteristics.

Run: python -m backend.tests.fixtures.generate_fixtures
     (from the repo root)

All fixtures are seeded for reproducibility. They are designed to exercise
senior-grade ML paths: imbalance, skewness, missing values, drift, multiclass.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

OUTPUT_DIR = __file__.replace("generate_fixtures.py", "")


def make_telco_churn(n: int = 7032) -> pd.DataFrame:
    """Binary classification, imbalanced (~14% churn)."""
    tenure = rng.integers(1, 72, n)
    monthly_charges = rng.normal(65, 30, n).clip(18, 120)
    total_charges = tenure * monthly_charges * rng.uniform(0.9, 1.1, n)
    num_products = rng.integers(1, 6, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n,
                          p=[0.55, 0.24, 0.21])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    tech_support = rng.choice(["Yes", "No", "No internet service"], n, p=[0.3, 0.5, 0.2])

    # Missing values (~3%)
    monthly_charges_missing = monthly_charges.copy().astype(float)
    monthly_charges_missing[rng.choice(n, int(0.03 * n), replace=False)] = np.nan

    # Churn: logistic-like probability
    logit = (
        -4.0
        + 0.04 * monthly_charges
        - 0.05 * tenure
        + 1.2 * (contract == "Month-to-month")
        + 0.5 * (internet_service == "Fiber optic")
        + 0.3 * senior
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(size=n) < prob).astype(int)

    return pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "tenure": tenure,
        "MonthlyCharges": monthly_charges_missing,
        "TotalCharges": total_charges,
        "NumProducts": num_products,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "InternetService": internet_service,
        "SeniorCitizen": senior,
        "TechSupport": tech_support,
        "Churn": churn,
    })


def make_credit_default(n: int = 30000) -> pd.DataFrame:
    """Binary classification with costly false negatives."""
    age = rng.integers(21, 79, n)
    limit_bal = rng.choice([10000, 20000, 50000, 100000, 200000], n)
    sex = rng.choice([1, 2], n)  # 1=male, 2=female
    education = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.47, 0.16, 0.02])
    marriage = rng.choice([1, 2, 3], n, p=[0.45, 0.53, 0.02])
    pay_0 = rng.integers(-2, 8, n)
    pay_amt1 = rng.lognormal(7, 2, n).clip(0, 1_000_000)
    bill_amt1 = rng.normal(51000, 73000, n).clip(0, 1_000_000)

    logit = (
        -3.5
        + 0.003 * (pay_0.clip(0) * limit_bal / 100000)
        - 0.01 * age
        + 0.3 * (pay_0 > 2)
        - 0.5 * (limit_bal > 100000)
    )
    prob = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(size=n) < prob).astype(int)

    return pd.DataFrame({
        "ID": range(1, n + 1),
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_AMT1": pay_amt1,
        "BILL_AMT1": bill_amt1,
        "default.payment.next.month": default,
    })


def make_lead_scoring(n: int = 9240) -> pd.DataFrame:
    """Binary classification, calibration-sensitive."""
    lead_origin = rng.choice(
        ["API", "Landing Page Submission", "Lead Add Form", "Lead Import"], n,
        p=[0.05, 0.38, 0.25, 0.32]
    )
    total_visits = rng.integers(0, 50, n)
    time_on_site = rng.lognormal(4, 1.5, n).clip(0, 3600)
    page_views = rng.integers(0, 40, n)
    last_activity = rng.choice(
        ["Email Opened", "SMS Sent", "Olark Chat Conversation", "Converted to Lead"], n
    )
    specialization = rng.choice(
        ["Business Administration", "Finance Management", "Human Resource Management",
         "Marketing Management", "Operations Management"], n
    )
    what_matters_most = rng.choice(
        ["Better Career Prospects", "Flexibility & Specialization", "Other"], n,
        p=[0.65, 0.25, 0.10]
    )
    city = rng.choice(["Mumbai", "Delhi", "Bangalore", "Other"], n, p=[0.5, 0.2, 0.2, 0.1])

    # Calibration: true probability is smooth
    logit = (
        -1.5
        + 0.02 * total_visits
        + 0.0003 * time_on_site
        + 0.03 * page_views
        + 0.8 * (what_matters_most == "Better Career Prospects")
        + 0.4 * (city == "Mumbai")
    )
    prob = 1 / (1 + np.exp(-logit))
    converted = (rng.uniform(size=n) < prob).astype(int)

    # Add some noise to make calibration non-trivial
    time_missing = time_on_site.copy().astype(float)
    time_missing[rng.choice(n, int(0.05 * n), replace=False)] = np.nan

    return pd.DataFrame({
        "Lead_Origin": lead_origin,
        "Total_Visits": total_visits,
        "Total_Time_Spent_on_Website": time_missing,
        "Page_Views_Per_Visit": page_views,
        "Last_Activity": last_activity,
        "Specialization": specialization,
        "What_matters_most_to_you_in_choosing_a_course": what_matters_most,
        "City": city,
        "Converted": converted,
    })


def make_claims_triage(n: int = 4500) -> pd.DataFrame:
    """Multiclass (4 triage levels): low, medium, high, critical."""
    claim_type = rng.choice(
        ["Auto", "Property", "Medical", "Liability"], n, p=[0.35, 0.30, 0.25, 0.10]
    )
    claim_amount = rng.lognormal(8, 1.5, n).clip(100, 5_000_000)
    claimant_age = rng.integers(18, 85, n)
    days_since_incident = rng.integers(0, 365, n)
    prior_claims = rng.integers(0, 10, n)
    fraud_score = rng.beta(1, 9, n)
    has_lawyer = rng.choice([0, 1], n, p=[0.7, 0.3])
    state = rng.choice(["CA", "TX", "FL", "NY", "WA"], n)

    # Triage level
    level = np.zeros(n, dtype=int)
    level[(claim_amount > 500_000) | (fraud_score > 0.5)] = 3  # critical
    level[(claim_amount > 100_000) & (level == 0)] = 2  # high
    level[(claim_amount > 20_000) & (level == 0)] = 1  # medium
    triage = pd.Categorical.from_codes(level, ["low", "medium", "high", "critical"])

    return pd.DataFrame({
        "claim_id": [f"CLM-{i:06d}" for i in range(n)],
        "claim_type": claim_type,
        "claim_amount": claim_amount,
        "claimant_age": claimant_age,
        "days_since_incident": days_since_incident,
        "prior_claims": prior_claims,
        "fraud_score": fraud_score,
        "has_lawyer": has_lawyer,
        "state": state,
        "triage_level": triage,
    })


def make_housing(n: int = 1460) -> pd.DataFrame:
    """Regression: house sale prices (log-transform decision)."""
    lot_area = rng.lognormal(9, 0.5, n).clip(1500, 200_000).astype(int)
    year_built = rng.integers(1900, 2010, n)
    overall_qual = rng.integers(1, 10, n)
    gr_liv_area = rng.normal(1500, 500, n).clip(300, 5000).astype(int)
    total_bsmt_sf = rng.normal(1000, 400, n).clip(0, 4000).astype(int)
    garage_cars = rng.integers(0, 4, n)
    neighborhood = rng.choice(
        ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
         "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW"], n
    )
    bldg_type = rng.choice(["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"], n,
                            p=[0.85, 0.04, 0.04, 0.05, 0.02])
    house_style = rng.choice(
        ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"], n,
        p=[0.50, 0.30, 0.10, 0.06, 0.04]
    )

    # Missing: basement and garage for some
    total_bsmt_sf_f = total_bsmt_sf.astype(float)
    total_bsmt_sf_f[rng.choice(n, int(0.02 * n), replace=False)] = np.nan

    # Sale price (log-normal with strong signal)
    log_price = (
        11.0
        + 0.1 * overall_qual
        + 0.0003 * gr_liv_area
        + 0.0001 * lot_area
        - 0.003 * (2023 - year_built)
        + rng.normal(0, 0.12, n)
    )
    sale_price = np.exp(log_price).clip(50_000, 800_000).astype(int)

    return pd.DataFrame({
        "LotArea": lot_area,
        "YearBuilt": year_built,
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf_f,
        "GarageCars": garage_cars,
        "Neighborhood": neighborhood,
        "BldgType": bldg_type,
        "HouseStyle": house_style,
        "SalePrice": sale_price,
    })


def make_telco_drift(n: int = 2000) -> pd.DataFrame:
    """Inference-role dataset with distributional drift from telco_churn."""
    rng2 = np.random.default_rng(99)
    tenure = rng2.integers(1, 36, n)  # shorter tenure (drift: new customer cohort)
    monthly_charges = rng2.normal(85, 25, n).clip(25, 120)  # higher charges (drift)
    total_charges = tenure * monthly_charges
    num_products = rng2.integers(1, 4, n)
    contract = rng2.choice(["Month-to-month", "One year", "Two year"], n,
                            p=[0.75, 0.18, 0.07])  # more month-to-month
    payment_method = rng2.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )
    internet_service = rng2.choice(["DSL", "Fiber optic", "No"], n, p=[0.20, 0.70, 0.10])
    senior = rng2.choice([0, 1], n, p=[0.75, 0.25])
    tech_support = rng2.choice(["Yes", "No", "No internet service"], n, p=[0.25, 0.60, 0.15])

    return pd.DataFrame({
        "customerID": [f"CUST-D-{i:05d}" for i in range(n)],
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "NumProducts": num_products,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "InternetService": internet_service,
        "SeniorCitizen": senior,
        "TechSupport": tech_support,
    })


def make_telco_holdout(n: int = 1000) -> pd.DataFrame:
    """Holdout dataset - same distribution as telco_churn but with labels."""
    rng3 = np.random.default_rng(777)
    base = make_telco_churn(n)
    # Reset the RNG-dependent random customerID
    base["customerID"] = [f"CUST-H-{i:05d}" for i in range(n)]
    return base


def make_patient_readmission(n: int = 5000) -> pd.DataFrame:
    """Binary classification - 30-day hospital readmission (imbalanced ~18%).

    Clinical context: predicting which discharged patients return within 30 days.
    High FN cost: missed readmissions lead to preventable deterioration.

    Exercises: SMOTE/class_weight, threshold optimisation (FN >> FP cost),
    SHAP interpretability of clinical variables, fairness across insurance type.
    """
    rng_r = np.random.default_rng(101)

    age = rng_r.integers(30, 90, n)
    gender = rng_r.choice(["M", "F"], n, p=[0.48, 0.52])
    length_of_stay = rng_r.integers(1, 30, n)  # days in hospital
    num_previous_admissions = rng_r.integers(0, 10, n)
    num_diagnoses = rng_r.integers(1, 15, n)
    num_procedures = rng_r.integers(0, 12, n)
    num_medications = rng_r.integers(1, 25, n)
    discharge_disposition = rng_r.choice(
        ["Home", "Skilled Nursing Facility", "Home Health Agency", "Rehab Facility"],
        n, p=[0.55, 0.20, 0.15, 0.10]
    )
    admission_type = rng_r.choice(["Emergency", "Urgent", "Elective"], n, p=[0.45, 0.35, 0.20])
    insurance_type = rng_r.choice(
        ["Medicare", "Medicaid", "Private", "Self-Pay"], n, p=[0.40, 0.25, 0.28, 0.07]
    )
    # Primary diagnosis chapter (ICD-10-style groups)
    primary_diagnosis = rng_r.choice(
        ["Circulatory", "Endocrine", "Respiratory", "Injury", "Digestive"],
        n, p=[0.28, 0.22, 0.18, 0.17, 0.15]
    )

    # Lab values with clinical reference ranges
    hba1c = rng_r.normal(7.2, 1.8, n).clip(4.0, 14.0)         # % - diabetic range ~6.5+
    creatinine = rng_r.lognormal(-0.1, 0.5, n).clip(0.5, 10.0) # mg/dL - elevated > 1.2
    sodium = rng_r.normal(139, 4, n).clip(120, 155)             # mEq/L - normal 136-145
    hemoglobin = rng_r.normal(12.5, 2.2, n).clip(6.0, 18.0)    # g/dL - low < 12 (F) / 13 (M)

    # Missing: lab values missing for ~8% of rows (not always drawn)
    hba1c_m = hba1c.copy().astype(float)
    hba1c_m[rng_r.choice(n, int(0.08 * n), replace=False)] = np.nan
    creatinine_m = creatinine.copy().astype(float)
    creatinine_m[rng_r.choice(n, int(0.06 * n), replace=False)] = np.nan

    # Readmission probability - clinically motivated (target ~18% positive rate)
    logit = (
        -3.9
        + 0.015 * age
        + 0.08 * num_previous_admissions
        + 0.04 * num_diagnoses
        + 0.03 * num_medications
        - 0.04 * length_of_stay.clip(0, 14)
        + 0.4 * (admission_type == "Emergency")
        + 0.5 * (primary_diagnosis == "Circulatory")
        + 0.3 * (primary_diagnosis == "Endocrine")
        + 0.4 * (np.nan_to_num(creatinine_m) > 1.5)
        + 0.35 * (np.nan_to_num(hba1c_m) > 8.0)
        + 0.3 * (insurance_type == "Medicaid")
        + 0.25 * (insurance_type == "Self-Pay")
        + rng_r.normal(0, 0.3, n)  # individual-level noise
    )
    prob = 1 / (1 + np.exp(-logit))
    readmitted = (rng_r.uniform(size=n) < prob).astype(int)

    return pd.DataFrame({
        "patient_id": [f"PAT-{i:06d}" for i in range(n)],
        "age": age,
        "gender": gender,
        "insurance_type": insurance_type,
        "admission_type": admission_type,
        "discharge_disposition": discharge_disposition,
        "primary_diagnosis_chapter": primary_diagnosis,
        "length_of_stay_days": length_of_stay,
        "num_previous_admissions": num_previous_admissions,
        "num_diagnoses": num_diagnoses,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "hba1c": hba1c_m,
        "creatinine": creatinine_m,
        "sodium": sodium,
        "hemoglobin": hemoglobin,
        "readmitted_30d": readmitted,
    })


def make_icu_mortality(n: int = 3000) -> pd.DataFrame:
    """Binary classification - ICU 48-hour mortality risk (costly FN, ~12% mortality).

    Clinical context: early warning system in ICU. Missing a patient at risk of
    dying (FN) is catastrophic; FN cost >> FP cost in this scenario.

    Exercises: extreme FN/FP cost ratio in threshold optimiser, tight calibration
    requirement (Brier < 0.10 is the clinical bar), SHAP for bedside explanation.
    """
    rng_i = np.random.default_rng(202)

    age = rng_i.integers(18, 90, n)
    gender = rng_i.choice(["M", "F"], n, p=[0.55, 0.45])
    icu_type = rng_i.choice(["Medical", "Surgical", "Cardiac", "Neuro"], n, p=[0.40, 0.30, 0.20, 0.10])

    # SOFA score components (Sequential Organ Failure Assessment)
    gcs = rng_i.integers(3, 15, n)                                  # Glasgow Coma Scale (lower = worse)
    pao2_fio2 = rng_i.normal(280, 120, n).clip(60, 500)             # respiratory (lower = worse)
    bilirubin = rng_i.lognormal(0.5, 0.9, n).clip(0.1, 30.0)       # liver (higher = worse)
    creatinine = rng_i.lognormal(0.2, 0.7, n).clip(0.5, 15.0)      # renal
    platelets = rng_i.normal(210, 90, n).clip(10, 600)              # coagulation
    map_bp = rng_i.normal(80, 20, n).clip(40, 140)                  # mean arterial pressure
    heart_rate = rng_i.normal(88, 22, n).clip(40, 180)
    respiratory_rate = rng_i.normal(18, 7, n).clip(8, 50)
    temperature = rng_i.normal(37.1, 0.9, n).clip(34.0, 41.0)
    spo2 = rng_i.normal(96, 4, n).clip(70, 100)
    urine_output_ml = rng_i.lognormal(6.5, 0.7, n).clip(50, 5000)  # 24h urine

    # Missing: vitals sometimes not yet recorded on admission
    pao2_fio2_m = pao2_fio2.copy().astype(float)
    pao2_fio2_m[rng_i.choice(n, int(0.12 * n), replace=False)] = np.nan
    urine_output_m = urine_output_ml.copy().astype(float)
    urine_output_m[rng_i.choice(n, int(0.07 * n), replace=False)] = np.nan

    # Mortality - all features normalised to [0,1] range so intercept is the floor.
    # Expected logit ≈ -2.0 → sigmoid(-2.0) ≈ 12% mortality.
    logit = (
        -2.0
        + 0.6 * (15 - gcs) / 12            # GCS 3=worst→1, 15=best→0
        - 0.5 * pao2_fio2 / 500            # PaO2/FiO2 500=best, 60=worst
        + 0.5 * bilirubin.clip(0, 20) / 20
        + 0.4 * creatinine.clip(0, 12) / 12
        - 0.3 * platelets.clip(10, 600) / 600
        - 0.3 * map_bp.clip(40, 140) / 140
        + 0.05 * age / 90
        + 0.4 * (icu_type == "Medical")
        - 0.0001 * np.nan_to_num(urine_output_ml)
        + rng_i.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    died_48h = (rng_i.uniform(size=n) < prob).astype(int)

    return pd.DataFrame({
        "patient_id": [f"ICU-{i:06d}" for i in range(n)],
        "age": age,
        "gender": gender,
        "icu_type": icu_type,
        "gcs_score": gcs,
        "pao2_fio2_ratio": pao2_fio2_m,
        "bilirubin_mgdl": bilirubin,
        "creatinine_mgdl": creatinine,
        "platelet_count": platelets,
        "mean_arterial_pressure": map_bp,
        "heart_rate_bpm": heart_rate,
        "respiratory_rate": respiratory_rate,
        "temperature_celsius": temperature,
        "spo2_pct": spo2,
        "urine_output_24h_ml": urine_output_m,
        "mortality_48h": died_48h,
    })


def make_disease_screening(n: int = 8000) -> pd.DataFrame:
    """Binary classification - diabetes screening (calibration-critical, ~15% prevalence).

    Clinical context: population-level screening program. Calibrated probabilities
    matter - clinicians use them to stratify intervention intensity.
    Platt / Isotonic calibration path will be exercised.

    Exercises: calibration (Brier score target < 0.12), PHI column detection
    (patient_name, dob), fairness across age group and gender.
    """
    rng_d = np.random.default_rng(303)

    # Deliberately include PHI-looking columns to test PHI detector (Phase 2)
    patient_name = [f"Patient {i}" for i in range(n)]  # fake names - triggers PHI flag
    dob_year = rng_d.integers(1940, 2000, n)
    dob = [f"{y}-{rng_d.integers(1,13):02d}-{rng_d.integers(1,29):02d}" for y in dob_year]
    mrn = [f"MRN{i:07d}" for i in range(n)]            # medical record number - triggers PHI flag

    age = 2024 - dob_year
    gender = rng_d.choice(["M", "F", "Other"], n, p=[0.49, 0.49, 0.02])
    ethnicity = rng_d.choice(
        ["White", "Hispanic", "Black", "Asian", "Other"], n,
        p=[0.60, 0.18, 0.13, 0.06, 0.03]
    )
    bmi = rng_d.normal(27.5, 6.5, n).clip(14, 60)
    systolic_bp = rng_d.normal(125, 18, n).clip(80, 200)
    diastolic_bp = rng_d.normal(80, 12, n).clip(50, 120)
    fasting_glucose = rng_d.normal(98, 28, n).clip(60, 400)   # mg/dL - IFG ≥ 100, DM ≥ 126
    hba1c_proxy = rng_d.normal(5.8, 1.2, n).clip(4.0, 12.0)  # % - pre-DM 5.7-6.4, DM ≥ 6.5
    hdl_cholesterol = rng_d.normal(52, 16, n).clip(20, 120)
    triglycerides = rng_d.lognormal(4.9, 0.6, n).clip(50, 800)
    family_history_diabetes = rng_d.choice([0, 1], n, p=[0.70, 0.30])
    physical_activity_days_wk = rng_d.integers(0, 7, n)
    smoking_status = rng_d.choice(["Never", "Former", "Current"], n, p=[0.55, 0.30, 0.15])

    # Missing: HbA1c proxy and fasting glucose not always tested
    hba1c_m = hba1c_proxy.copy().astype(float)
    hba1c_m[rng_d.choice(n, int(0.10 * n), replace=False)] = np.nan
    glucose_m = fasting_glucose.copy().astype(float)
    glucose_m[rng_d.choice(n, int(0.08 * n), replace=False)] = np.nan

    # Diabetes probability - based on known risk factors (target ~15% prevalence)
    logit = (
        -3.5
        + 0.025 * (age - 40).clip(0)
        + 0.08 * (bmi - 25).clip(0)
        + 0.02 * (fasting_glucose - 100).clip(0)
        + 1.2 * (hba1c_proxy > 5.7)
        + 0.7 * family_history_diabetes
        + 0.2 * (smoking_status == "Current")
        - 0.08 * physical_activity_days_wk
        + 0.35 * (ethnicity == "Hispanic")
        + 0.25 * (ethnicity == "Black")
        + rng_d.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    diabetes_dx = (rng_d.uniform(size=n) < prob).astype(int)

    return pd.DataFrame({
        "patient_name": patient_name,  # PHI - should be flagged
        "mrn": mrn,                    # PHI - should be flagged
        "date_of_birth": dob,          # PHI - should be flagged
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "fasting_glucose_mgdl": glucose_m,
        "hba1c_pct": hba1c_m,
        "hdl_cholesterol_mgdl": hdl_cholesterol,
        "triglycerides_mgdl": triglycerides,
        "family_history_diabetes": family_history_diabetes,
        "physical_activity_days_per_week": physical_activity_days_wk,
        "smoking_status": smoking_status,
        "diabetes_diagnosis": diabetes_dx,
    })


def make_diagnosis_triage(n: int = 2000) -> pd.DataFrame:
    """Multiclass (4 diagnosis categories) - ED triage chief complaint.

    Clinical context: emergency department triage - categorise chief complaint
    into 4 diagnosis groups for routing. Intersectional fairness analysis
    across gender × insurance_type.

    Exercises: multiclass classification, fairness across 2 protected attributes,
    per-class SHAP, intersectional equity report.
    """
    rng_t = np.random.default_rng(404)

    age = rng_t.integers(18, 85, n)
    gender = rng_t.choice(["M", "F", "Other"], n, p=[0.47, 0.51, 0.02])
    insurance_type = rng_t.choice(
        ["Medicare", "Medicaid", "Private", "Self-Pay"], n, p=[0.30, 0.28, 0.32, 0.10]
    )
    arrival_mode = rng_t.choice(["Ambulance", "Walk-in", "Transfer"], n, p=[0.35, 0.55, 0.10])
    triage_level = rng_t.integers(1, 6, n)  # ESI 1-5 (1 = most urgent)

    # Vital signs at triage
    systolic_bp = rng_t.normal(130, 22, n).clip(70, 220)
    heart_rate = rng_t.normal(85, 22, n).clip(35, 180)
    respiratory_rate = rng_t.normal(17, 5, n).clip(8, 45)
    temperature = rng_t.normal(37.0, 1.0, n).clip(34.5, 41.5)
    spo2 = rng_t.normal(96, 5, n).clip(70, 100)
    pain_score = rng_t.integers(0, 11, n)

    # Chief complaint keywords (ordinal encoded)
    chest_pain = rng_t.choice([0, 1], n, p=[0.82, 0.18])
    dyspnea = rng_t.choice([0, 1], n, p=[0.80, 0.20])
    altered_mental_status = rng_t.choice([0, 1], n, p=[0.90, 0.10])
    abdominal_pain = rng_t.choice([0, 1], n, p=[0.75, 0.25])
    trauma = rng_t.choice([0, 1], n, p=[0.85, 0.15])
    fever = (temperature > 38.3).astype(int)

    # Missing: SpO2 not always recorded before triage room
    spo2_m = spo2.copy().astype(float)
    spo2_m[rng_t.choice(n, int(0.05 * n), replace=False)] = np.nan

    # Diagnosis category assignment - clinically motivated
    # 0=Cardiac, 1=Respiratory, 2=Trauma/Surgical, 3=Infectious/Other
    score_cardiac = chest_pain * 3 + (heart_rate > 100).astype(int) + (systolic_bp < 90).astype(int) * 2
    score_respiratory = dyspnea * 3 + (respiratory_rate > 24).astype(int) * 2 + (spo2 < 94).astype(int) * 2
    score_trauma = trauma * 4 + abdominal_pain + pain_score / 5
    score_infectious = fever * 3 + altered_mental_status * 2 + abdominal_pain

    scores = np.stack([score_cardiac, score_respiratory, score_trauma, score_infectious], axis=1)
    scores += rng_t.normal(0, 1, scores.shape)  # add noise
    category_idx = scores.argmax(axis=1)
    category_labels = np.array(["Cardiac", "Respiratory", "Trauma_Surgical", "Infectious_Other"])
    diagnosis_category = category_labels[category_idx]

    return pd.DataFrame({
        "encounter_id": [f"ED-{i:06d}" for i in range(n)],
        "age": age,
        "gender": gender,
        "insurance_type": insurance_type,
        "arrival_mode": arrival_mode,
        "triage_level_esi": triage_level,
        "systolic_bp": systolic_bp,
        "heart_rate_bpm": heart_rate,
        "respiratory_rate": respiratory_rate,
        "temperature_celsius": temperature,
        "spo2_pct": spo2_m,
        "pain_score": pain_score,
        "chief_complaint_chest_pain": chest_pain,
        "chief_complaint_dyspnea": dyspnea,
        "chief_complaint_altered_ms": altered_mental_status,
        "chief_complaint_abdominal": abdominal_pain,
        "chief_complaint_trauma": trauma,
        "diagnosis_category": diagnosis_category,
    })


if __name__ == "__main__":
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating test fixtures…")
    make_telco_churn().to_csv(f"{OUTPUT_DIR}/telco_churn.csv", index=False)
    print("  telco_churn.csv (7032 rows)")

    make_credit_default().to_csv(f"{OUTPUT_DIR}/credit_default.csv", index=False)
    print("  credit_default.csv (30000 rows)")

    make_lead_scoring().to_csv(f"{OUTPUT_DIR}/lead_scoring.csv", index=False)
    print("  lead_scoring.csv (9240 rows)")

    make_claims_triage().to_csv(f"{OUTPUT_DIR}/claims_triage.csv", index=False)
    print("  claims_triage.csv (4500 rows)")

    make_housing().to_csv(f"{OUTPUT_DIR}/housing.csv", index=False)
    print("  housing.csv (1460 rows)")

    make_telco_drift().to_csv(f"{OUTPUT_DIR}/telco_churn_drift_2025.csv", index=False)
    print("  telco_churn_drift_2025.csv (2000 rows)")

    make_telco_holdout().to_csv(f"{OUTPUT_DIR}/telco_churn_holdout.csv", index=False)
    print("  telco_churn_holdout.csv (1000 rows)")

    make_patient_readmission().to_csv(f"{OUTPUT_DIR}/patient_readmission.csv", index=False)
    print("  patient_readmission.csv (5000 rows) - 30-day readmission, imbalanced ~18%")

    make_icu_mortality().to_csv(f"{OUTPUT_DIR}/icu_mortality.csv", index=False)
    print("  icu_mortality.csv (3000 rows) - 48h ICU mortality, costly FN ~12%")

    make_disease_screening().to_csv(f"{OUTPUT_DIR}/disease_screening.csv", index=False)
    print("  disease_screening.csv (8000 rows) - diabetes screening, PHI columns included")

    make_diagnosis_triage().to_csv(f"{OUTPUT_DIR}/diagnosis_triage.csv", index=False)
    print("  diagnosis_triage.csv (2000 rows) - multiclass ED triage, intersectional fairness")

    print("Done.")
