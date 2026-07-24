"""
templates.py
------------
Generates the final human-readable disease report by randomly picking one
of 10 differently-formatted templates. Logic ported as-is from the
original notebook.
"""

import random


def language_output(retrieved_info):
    """
    Generates a randomly formatted disease report from one of 10 templates.

    Input:
        retrieved_info (dict): Disease information dictionary.

    Returns:
        str: Formatted report.
    """

    disease_name = retrieved_info.get("display_name", retrieved_info.get("original_class_label", "Unknown"))

    confidence = retrieved_info.get("predicted_probability", "N/A")
    if isinstance(confidence, float):
        confidence = f"{confidence:.2%}"

    templates = {
        1: f"""
Crop: {retrieved_info['crop']}

Disease: {disease_name}

Confidence: {confidence}

Symptoms:
{retrieved_info['symptoms']}

Cause:
{retrieved_info['cause']}

Organic Treatment:
{retrieved_info['organic_treatment']}

Chemical Treatment:
{retrieved_info['chemical_treatment']}

Prevention:
{retrieved_info['prevention']}

Severity:
{retrieved_info['severity']}

Source:
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        2: f"""
========== DISEASE REPORT ==========

Crop              : {retrieved_info['crop']}
Disease           : {disease_name}
Prediction Score  : {confidence}

What you may observe:
{retrieved_info['symptoms']}

Likely cause:
{retrieved_info['cause']}

Organic control:
{retrieved_info['organic_treatment']}

Chemical control:
{retrieved_info['chemical_treatment']}

Prevention:
{retrieved_info['prevention']}

Severity Level:
{retrieved_info['severity']}

Reference:
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        3: f"""
Crop Identified: {retrieved_info['crop']}

Detected Disease: {disease_name}
Model Confidence: {confidence}

Disease Summary
---------------
Symptoms:
{retrieved_info['symptoms']}

Cause:
{retrieved_info['cause']}

Recommended Organic Treatment:
{retrieved_info['organic_treatment']}

Recommended Chemical Treatment:
{retrieved_info['chemical_treatment']}

Prevention Tips:
{retrieved_info['prevention']}

Severity:
{retrieved_info['severity']}

Information Source:
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        4: f"""
PLANT HEALTH REPORT

Crop: {retrieved_info['crop']}
Disease: {disease_name}
Confidence: {confidence}

Symptoms Observed:
• {retrieved_info['symptoms']}

Possible Cause:
• {retrieved_info['cause']}

Organic Management:
• {retrieved_info['organic_treatment']}

Chemical Management:
• {retrieved_info['chemical_treatment']}

Preventive Measures:
• {retrieved_info['prevention']}

Severity: {retrieved_info['severity']}

Reference:
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        5: f"""
Disease Detection Result

Crop              : {retrieved_info['crop']}
Disease           : {disease_name}
Confidence Score  : {confidence}

Symptoms:
{retrieved_info['symptoms']}

Cause:
{retrieved_info['cause']}

Recommended Treatments
----------------------
Organic:
{retrieved_info['organic_treatment']}

Chemical:
{retrieved_info['chemical_treatment']}

Prevention:
{retrieved_info['prevention']}

Severity Rating:
{retrieved_info['severity']}

Reference:
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        6: f"""
AI Crop Diagnosis

Crop: {retrieved_info['crop']}
Predicted Disease: {disease_name}
Confidence: {confidence}

Symptoms
---------
{retrieved_info['symptoms']}

Cause
-----
{retrieved_info['cause']}

Organic Remedy
--------------
{retrieved_info['organic_treatment']}

Chemical Remedy
---------------
{retrieved_info['chemical_treatment']}

Prevent Future Infection
------------------------
{retrieved_info['prevention']}

Severity:
{retrieved_info['severity']}

Source:
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        7: f"""
Crop Health Assessment

Crop: {retrieved_info['crop']}

Disease Prediction:
{disease_name}

Prediction Confidence:
{confidence}

Symptoms:
{retrieved_info['symptoms']}

Reason:
{retrieved_info['cause']}

Organic Treatment:
{retrieved_info['organic_treatment']}

Chemical Treatment:
{retrieved_info['chemical_treatment']}

Prevention Strategy:
{retrieved_info['prevention']}

Disease Severity:
{retrieved_info['severity']}

Reference:
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        8: f"""
Prediction Summary

Crop: {retrieved_info['crop']}
Disease: {disease_name}
Confidence: {confidence}

Observed Symptoms
{retrieved_info['symptoms']}

Possible Cause
{retrieved_info['cause']}

Organic Control
{retrieved_info['organic_treatment']}

Chemical Control
{retrieved_info['chemical_treatment']}

Prevention Advice
{retrieved_info['prevention']}

Severity Level
{retrieved_info['severity']}

Source Information
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        9: f"""
Disease Analysis Report

Crop Name:
{retrieved_info['crop']}

Disease Name:
{disease_name}

Confidence:
{confidence}

Symptoms:
{retrieved_info['symptoms']}

Cause:
{retrieved_info['cause']}

Organic Solution:
{retrieved_info['organic_treatment']}

Chemical Solution:
{retrieved_info['chemical_treatment']}

How to Prevent:
{retrieved_info['prevention']}

Severity:
{retrieved_info['severity']}

Reference:
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
        10: f"""
Crop Disease Detection

Crop                : {retrieved_info['crop']}
Disease             : {disease_name}
Confidence          : {confidence}

Symptoms            : {retrieved_info['symptoms']}

Cause               : {retrieved_info['cause']}

Organic Treatment   : {retrieved_info['organic_treatment']}

Chemical Treatment  : {retrieved_info['chemical_treatment']}

Prevention          : {retrieved_info['prevention']}

Severity            : {retrieved_info['severity']}

Reference           :
{retrieved_info['source']}
{retrieved_info['source_url']}
""",
    }

    template_id = random.randint(1, 10)

    return templates[template_id]
