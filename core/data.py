"""
Hospital Department Profiles, Visual Specialty Prompts, and Simulated Doctor Database.

DISCLAIMER:
All physician profiles in DOCTORS_DB are synthetic mock data generated strictly for
research, demonstration, and evaluation purposes in the Stat-a-thon course project.
They do not represent real medical staff at CMC Vellore or any other healthcare facility.
This system is an administrative routing decision-support aid, not a clinical diagnostic device.
"""

from typing import Dict, List, Any

# 8 Multispecialty Hospital Departments & Symptom Profiles
DEPARTMENT_SYMPTOM_PROFILES: Dict[str, str] = {
    "General Dermatology & Cosmetology": (
        "skin rash, itch, eczema, nodules, red patches, psoriasis, acne, pimples, "
        "blisters, pustules, hair loss, alopecia, nail discoloration, vitiligo, "
        "skin lesions, ulcers, fungal infection, warts, dermatitis, mucosal problems, hives"
    ),
    "Cardiology & Cardiothoracic": (
        "chest pain, chest tightness, heart palpitations, irregular heartbeat, "
        "shortness of breath on exertion, high blood pressure, hypertension, "
        "dizziness, radiating pain in left arm or jaw, swelling in legs, angina, tachycardia"
    ),
    "Orthopaedics & Rheumatology": (
        "joint pain, knee swelling, bone fracture, back pain, neck stiffness, "
        "slipped disc, ligament tear, sprain, arthritis, shoulder dislocation, "
        "difficulty walking, sports injury, rheumatoid, spondylitis"
    ),
    "Neurology & Neurosurgery": (
        "severe headache, migraine, seizures, fits, numbness, tingling sensation, "
        "muscle weakness, paralysis, stroke, tremors, memory loss, vertigo, "
        "loss of balance, slurred speech, nerve pain"
    ),
    "Ophthalmology (Eye Care)": (
        "blurry vision, eye pain, red eyes, watery eyes, itchy eyes, cataract, "
        "glaucoma, double vision, sensitivity to light, eyelid swelling, floaters, conjunctivitis"
    ),
    "Gastroenterology & Hepatology": (
        "stomach pain, abdominal cramps, acid reflux, heartburn, indigestion, "
        "nausea, vomiting, diarrhea, constipation, bloating, blood in stool, "
        "ulcer, jaundice, liver problem, gastritis"
    ),
    "Pulmonology (Respiratory Medicine)": (
        "chronic cough, coughing up blood, wheezing, asthma, shortness of breath, "
        "breathing difficulty, chest congestion, phlegm, sputum, pneumonia, "
        "bronchitis, tuberculosis, sleep apnea"
    ),
    "ENT (Otorhinolaryngology)": (
        "ear pain, ear discharge, hearing loss, ringing in ears, tinnitus, "
        "sore throat, difficulty swallowing, tonsillitis, sinus congestion, "
        "runny nose, nasal bleeding, hoarse voice"
    ),
    "General Medicine (Internal Medicine)": (
        "fever, chills, body ache, fatigue, weakness, tiredness, loss of appetite, "
        "weight loss, general malaise, cold, flu, viral infection, headache with fever, "
        "dizziness, dehydration, diabetes, thyroid, anaemia, routine check-up, "
        "multiple unrelated symptoms, unwell"
    )
}

# Zero-Shot Prompts for Local BiomedCLIP (Vision)
VISUAL_SPECIALTY_PROMPTS: Dict[str, str] = {
    "General Dermatology & Cosmetology": "a clinical photograph of a skin rash, eczema, psoriasis, acne, skin lesion, blister, or nail disorder",
    "Cardiology & Cardiothoracic": "an electrocardiogram ECG chart, chest X-ray showing cardiomegaly, or pedal edema leg swelling",
    "Orthopaedics & Rheumatology": "a clinical photograph or X-ray of a swollen joint, bone fracture, knee injury, sprained ankle, or limb deformity",
    "Neurology & Neurosurgery": "a brain MRI or CT scan, facial palsy asymmetry, or neurological clinical presentation",
    "Ophthalmology (Eye Care)": "a clinical photograph of an eye condition, red eye, conjunctivitis, cataract, eyelid swelling, or corneal issue",
    "Gastroenterology & Hepatology": "an endoscopic image, abdominal scan, jaundice yellowing, or gastrointestinal clinical presentation",
    "Pulmonology (Respiratory Medicine)": "a chest X-ray radiograph or CT scan of lungs showing pneumonia, bronchitis, or pulmonary infiltrate",
    "ENT (Otorhinolaryngology)": "a clinical photograph of an inflamed throat, tonsillitis, oral cavity, ear infection, or nasal condition",
    "General Medicine (Internal Medicine)": "a clinical photograph showing general signs of illness such as pallor, generalised swelling, or dehydration, with no single organ-specific finding"
}

# Mock Hospital Doctors Database (2-3 specialists per department, 21 total)
# Clearly designated as SAMPLE / MOCK data
DOCTORS_DB: List[Dict[str, Any]] = [
    # 1. General Dermatology & Cosmetology
    {
        "id": "DOC101",
        "name": "Dr. Aravind Swaminathan (Mock)",
        "specialty": "General Dermatology & Cosmetology",
        "sub_focus": "eczema, psoriasis, autoimmune blistering, rash, red patches, scaly skin, dermatitis",
        "exp_years": 16,
        "rating": 4.9,
        "success_rate": 0.96,
        "patients_treated": 4200,
        "availability": "Available Today (10:00 AM)",
        "avail_score": 1.0,
        "is_mock": True
    },
    {
        "id": "DOC102",
        "name": "Dr. Meenakshi Sundaram (Mock)",
        "specialty": "General Dermatology & Cosmetology",
        "sub_focus": "acne, pimples, nail discoloration, hair loss, alopecia, mucosal, warts, fungal",
        "exp_years": 11,
        "rating": 4.7,
        "success_rate": 0.93,
        "patients_treated": 2850,
        "availability": "Available Today (2:00 PM)",
        "avail_score": 0.90,
        "is_mock": True
    },
    {
        "id": "DOC103",
        "name": "Dr. Preethi Raghavan (Mock)",
        "specialty": "General Dermatology & Cosmetology",
        "sub_focus": "vitiligo, hives, urticaria, skin ulcers, blisters, cosmetic lesions, pustules",
        "exp_years": 8,
        "rating": 4.6,
        "success_rate": 0.91,
        "patients_treated": 1900,
        "availability": "Available Tomorrow (11:00 AM)",
        "avail_score": 0.70,
        "is_mock": True
    },

    # 2. Cardiology & Cardiothoracic
    {
        "id": "DOC201",
        "name": "Dr. Vikramaditya Rao (Mock)",
        "specialty": "Cardiology & Cardiothoracic",
        "sub_focus": "chest pain, angina, coronary artery, left arm pain, heart attack, tightness",
        "exp_years": 22,
        "rating": 4.9,
        "success_rate": 0.97,
        "patients_treated": 6100,
        "availability": "Available Today (11:00 AM)",
        "avail_score": 1.0,
        "is_mock": True
    },
    {
        "id": "DOC202",
        "name": "Dr. Priya Nair (Mock)",
        "specialty": "Cardiology & Cardiothoracic",
        "sub_focus": "palpitations, arrhythmia, hypertension, blood pressure, dizziness, swelling",
        "exp_years": 13,
        "rating": 4.8,
        "success_rate": 0.94,
        "patients_treated": 3100,
        "availability": "Available Today (3:00 PM)",
        "avail_score": 0.85,
        "is_mock": True
    },
    {
        "id": "DOC203",
        "name": "Dr. Rajesh K. Varma (Mock)",
        "specialty": "Cardiology & Cardiothoracic",
        "sub_focus": "heart failure, shortness of breath, leg edema, valvular heart disease, tachycardia",
        "exp_years": 17,
        "rating": 4.8,
        "success_rate": 0.95,
        "patients_treated": 4300,
        "availability": "Available Tomorrow (9:00 AM)",
        "avail_score": 0.75,
        "is_mock": True
    },

    # 3. Orthopaedics & Rheumatology
    {
        "id": "DOC301",
        "name": "Dr. Senthil Kumar V (Mock)",
        "specialty": "Orthopaedics & Rheumatology",
        "sub_focus": "knee swelling, joint pain, ligament tear, fracture, sports injury, ankle sprain",
        "exp_years": 18,
        "rating": 4.8,
        "success_rate": 0.95,
        "patients_treated": 5300,
        "availability": "Available Today (9:30 AM)",
        "avail_score": 1.0,
        "is_mock": True
    },
    {
        "id": "DOC302",
        "name": "Dr. Ananya Banerjee (Mock)",
        "specialty": "Orthopaedics & Rheumatology",
        "sub_focus": "arthritis, back pain, neck stiffness, spondylitis, slipped disc, rheumatoid",
        "exp_years": 10,
        "rating": 4.7,
        "success_rate": 0.92,
        "patients_treated": 2400,
        "availability": "Available Today (4:00 PM)",
        "avail_score": 0.85,
        "is_mock": True
    },
    {
        "id": "DOC303",
        "name": "Dr. Tarun Nambiar (Mock)",
        "specialty": "Orthopaedics & Rheumatology",
        "sub_focus": "shoulder dislocation, difficulty walking, hip replacement, cartilage injury",
        "exp_years": 14,
        "rating": 4.8,
        "success_rate": 0.94,
        "patients_treated": 3700,
        "availability": "Available Today (1:00 PM)",
        "avail_score": 0.90,
        "is_mock": True
    },

    # 4. Neurology & Neurosurgery
    {
        "id": "DOC401",
        "name": "Dr. Karthikeya Murthy (Mock)",
        "specialty": "Neurology & Neurosurgery",
        "sub_focus": "migraine, headache, seizures, epilepsy, stroke, paralysis, numbness",
        "exp_years": 19,
        "rating": 4.9,
        "success_rate": 0.96,
        "patients_treated": 4800,
        "availability": "Available Today (10:30 AM)",
        "avail_score": 1.0,
        "is_mock": True
    },
    {
        "id": "DOC402",
        "name": "Dr. Shalini Deshmukh (Mock)",
        "specialty": "Neurology & Neurosurgery",
        "sub_focus": "tremors, parkinsons, vertigo, loss of balance, slurred speech, tingling sensation",
        "exp_years": 12,
        "rating": 4.7,
        "success_rate": 0.93,
        "patients_treated": 2900,
        "availability": "Available Today (2:30 PM)",
        "avail_score": 0.90,
        "is_mock": True
    },
    {
        "id": "DOC403",
        "name": "Dr. Arvind Chacko (Mock)",
        "specialty": "Neurology & Neurosurgery",
        "sub_focus": "memory loss, dementia, nerve pain, neuropathy, muscle weakness, fits",
        "exp_years": 15,
        "rating": 4.8,
        "success_rate": 0.94,
        "patients_treated": 3600,
        "availability": "Available Tomorrow (10:00 AM)",
        "avail_score": 0.70,
        "is_mock": True
    },

    # 5. Ophthalmology (Eye Care)
    {
        "id": "DOC501",
        "name": "Dr. Rajeshwari Menon (Mock)",
        "specialty": "Ophthalmology (Eye Care)",
        "sub_focus": "red eye, conjunctivitis, blurry vision, watery eyes, cataract, cornea, itchy eye",
        "exp_years": 15,
        "rating": 4.8,
        "success_rate": 0.95,
        "patients_treated": 3900,
        "availability": "Available Today (11:30 AM)",
        "avail_score": 1.0,
        "is_mock": True
    },
    {
        "id": "DOC502",
        "name": "Dr. Gautam Sen (Mock)",
        "specialty": "Ophthalmology (Eye Care)",
        "sub_focus": "glaucoma, double vision, eyelid swelling, floaters, light sensitivity, retina",
        "exp_years": 18,
        "rating": 4.9,
        "success_rate": 0.96,
        "patients_treated": 4700,
        "availability": "Available Today (3:30 PM)",
        "avail_score": 0.85,
        "is_mock": True
    },
    {
        "id": "DOC503",
        "name": "Dr. Divya Chandran (Mock)",
        "specialty": "Ophthalmology (Eye Care)",
        "sub_focus": "refractive error, dry eye, corneal ulcer, eye discharge, vision loss",
        "exp_years": 9,
        "rating": 4.6,
        "success_rate": 0.92,
        "patients_treated": 2100,
        "availability": "Available Tomorrow (2:00 PM)",
        "avail_score": 0.70,
        "is_mock": True
    },

    # 6. Gastroenterology & Hepatology
    {
        "id": "DOC601",
        "name": "Dr. Mohammed Farooq (Mock)",
        "specialty": "Gastroenterology & Hepatology",
        "sub_focus": "stomach pain, acid reflux, heartburn, vomiting, diarrhea, ulcer, jaundice, bloating",
        "exp_years": 17,
        "rating": 4.9,
        "success_rate": 0.95,
        "patients_treated": 4400,
        "availability": "Available Today (12:00 PM)",
        "avail_score": 0.95,
        "is_mock": True
    },
    {
        "id": "DOC602",
        "name": "Dr. Kavita Krishnan (Mock)",
        "specialty": "Gastroenterology & Hepatology",
        "sub_focus": "gastritis, indigestion, constipation, abdominal cramps, blood in stool, liver problem",
        "exp_years": 12,
        "rating": 4.7,
        "success_rate": 0.93,
        "patients_treated": 3000,
        "availability": "Available Today (4:30 PM)",
        "avail_score": 0.80,
        "is_mock": True
    },
    {
        "id": "DOC603",
        "name": "Dr. Anand Joshi (Mock)",
        "specialty": "Gastroenterology & Hepatology",
        "sub_focus": "cirrhosis, hepatitis, nausea, peptic ulcer, irritable bowel, gallbladder",
        "exp_years": 20,
        "rating": 4.9,
        "success_rate": 0.97,
        "patients_treated": 5200,
        "availability": "Available Tomorrow (11:30 AM)",
        "avail_score": 0.75,
        "is_mock": True
    },

    # 7. Pulmonology (Respiratory Medicine)
    {
        "id": "DOC701",
        "name": "Dr. Lakshmi Narayanan (Mock)",
        "specialty": "Pulmonology (Respiratory Medicine)",
        "sub_focus": "cough, asthma, wheezing, shortness of breath, phlegm, sputum, bronchitis",
        "exp_years": 14,
        "rating": 4.8,
        "success_rate": 0.94,
        "patients_treated": 3600,
        "availability": "Available Today (1:30 PM)",
        "avail_score": 0.90,
        "is_mock": True
    },
    {
        "id": "DOC702",
        "name": "Dr. Rohan Mukherjee (Mock)",
        "specialty": "Pulmonology (Respiratory Medicine)",
        "sub_focus": "pneumonia, tuberculosis, chest congestion, sleep apnea, chronic bronchitis, breathing difficulty",
        "exp_years": 19,
        "rating": 4.9,
        "success_rate": 0.96,
        "patients_treated": 4900,
        "availability": "Available Today (10:00 AM)",
        "avail_score": 1.0,
        "is_mock": True
    },
    {
        "id": "DOC703",
        "name": "Dr. Sneha Paul (Mock)",
        "specialty": "Pulmonology (Respiratory Medicine)",
        "sub_focus": "allergy, respiratory infection, hemoptysis, chronic cough, lung nodule",
        "exp_years": 8,
        "rating": 4.6,
        "success_rate": 0.91,
        "patients_treated": 1800,
        "availability": "Available Tomorrow (3:00 PM)",
        "avail_score": 0.70,
        "is_mock": True
    },

    # 8. ENT (Otorhinolaryngology)
    {
        "id": "DOC801",
        "name": "Dr. Philipose George (Mock)",
        "specialty": "ENT (Otorhinolaryngology)",
        "sub_focus": "ear pain, sore throat, tonsillitis, sinus, hearing loss, tinnitus, nasal bleeding",
        "exp_years": 15,
        "rating": 4.8,
        "success_rate": 0.94,
        "patients_treated": 3800,
        "availability": "Available Today (10:00 AM)",
        "avail_score": 1.0,
        "is_mock": True
    },
    {
        "id": "DOC802",
        "name": "Dr. Rashmi Iyer (Mock)",
        "specialty": "ENT (Otorhinolaryngology)",
        "sub_focus": "ear discharge, hoarse voice, sinus congestion, runny nose, difficulty swallowing, dizziness",
        "exp_years": 11,
        "rating": 4.7,
        "success_rate": 0.93,
        "patients_treated": 2700,
        "availability": "Available Today (2:00 PM)",
        "avail_score": 0.90,
        "is_mock": True
    },
    {
        "id": "DOC803",
        "name": "Dr. Bimal Das (Mock)",
        "specialty": "ENT (Otorhinolaryngology)",
        "sub_focus": "nasal septum, adenoids, vertigo, vocal cord, otitis media, throat infection",
        "exp_years": 21,
        "rating": 4.9,
        "success_rate": 0.96,
        "patients_treated": 5500,
        "availability": "Available Tomorrow (11:00 AM)",
        "avail_score": 0.75,
        "is_mock": True
    },

    # 9. General Medicine (Internal Medicine)
    {
        "id": "DOC901",
        "name": "Dr. Sundararajan Ramanathan (Mock)",
        "specialty": "General Medicine (Internal Medicine)",
        "sub_focus": "fever, chills, body ache, viral infection, dengue, malaria, fatigue, malaise",
        "exp_years": 20,
        "rating": 4.9,
        "success_rate": 0.96,
        "patients_treated": 5800,
        "availability": "Available Today (9:00 AM)",
        "avail_score": 1.0,
        "is_mock": True
    },
    {
        "id": "DOC902",
        "name": "Dr. Haripriya Natarajan (Mock)",
        "specialty": "General Medicine (Internal Medicine)",
        "sub_focus": "diabetes, thyroid, hypertension follow-up, anaemia, weakness, tiredness, routine check-up",
        "exp_years": 14,
        "rating": 4.8,
        "success_rate": 0.94,
        "patients_treated": 3900,
        "availability": "Available Today (11:30 AM)",
        "avail_score": 0.90,
        "is_mock": True
    },
    {
        "id": "DOC903",
        "name": "Dr. Abraham Joseph (Mock)",
        "specialty": "General Medicine (Internal Medicine)",
        "sub_focus": "unwell, cold, flu, dehydration, headache with fever, loss of appetite, multiple unrelated symptoms",
        "exp_years": 11,
        "rating": 4.7,
        "success_rate": 0.92,
        "patients_treated": 2600,
        "availability": "Available Tomorrow (10:00 AM)",
        "avail_score": 0.75,
        "is_mock": True
    }
]
