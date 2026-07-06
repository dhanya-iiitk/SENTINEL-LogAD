"""
SENTINEL Explainability — Validation Module
Evaluates explanation quality using:
  - Grounding Score (GS)
  - Consistency Rate (CR)
  - Cross-LLM Similarity (CS, TC, RC, OQ) via Mistral AI
  - Expert human evaluation (U, R)
"""
from mistralai import Mistral
import json

THREAT_WORDS = ['attack', 'threat', 'malicious', 'exploit',
                'breach', 'intrusion', 'compromise', 'adversar']


def grounding_score(explanation, top_events):
    """Fraction of top-k IG events referenced in explanation."""
    grounded = 0
    for e in top_events:
        terms = [w for w in e['template'].split()
                 if len(w) > 3 and (w.isupper() or w.isalpha())]
        if any(t.lower() in explanation.lower() for t in terms):
            grounded += 1
    return grounded / len(top_events) if top_events else 0.0


def consistency_check(explanation, predicted):
    """Check explanation tone matches predicted label."""
    claims_threat = any(w in explanation.lower()
                        for w in THREAT_WORDS)
    if predicted == 'Normal':
        return not claims_threat
    return True


def mistral_evaluate_similarity(explanations, top_events,
                                 predicted, mistral_api_key):
    """
    Mistral AI evaluates cross-LLM consistency across
    GPT-OSS-120B, LLaMA-4-Scout, and Qwen3-32B outputs.

    Returns scores for:
      contextual_similarity  : same anomaly context?
      threat_consistency     : same threat identified?
      remediation_consistency: same remediation suggested?
      overall_quality        : collectively actionable?
    """
    client     = Mistral(api_key=mistral_api_key.strip())
    events_str = '\n'.join([
        f"  {e['rank']}. [{e['template']}]"
        for e in top_events])

    llm1 = explanations.get('LLM1_GPT120B',  'N/A')
    llm2 = explanations.get('LLM2_LLaMA4',   'N/A')
    llm3 = explanations.get('LLM3_Qwen3',    'N/A')

    prompt = f"""You are an expert cybersecurity analyst
evaluating three independent explanations generated
by different AI models for the same log anomaly case.

Predicted label: {predicted}
Top contributing log events (Integrated Gradients):
{events_str}

Explanation from LLM1 (GPT-OSS-120B):
{llm1}

Explanation from LLM2 (LLaMA-4-Scout):
{llm2}

Explanation from LLM3 (Qwen3-32B):
{llm3}

Rate on 1-5 scale:
1. Contextual Similarity: same anomaly context?
2. Threat Consistency: same threat identified?
3. Remediation Consistency: same remediation?
4. Overall Quality: technically sound and actionable?

JSON only:
{{
  "contextual_similarity": X,
  "threat_consistency": X,
  "remediation_consistency": X,
  "overall_quality": X,
  "summary": "one sentence"
}}"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=250)
    text = response.choices[0].message.content.strip()
    text = text.replace('```json','').replace('```','').strip()
    return json.loads(text)


def mistral_evaluate(explanation, top_events, predicted,
                     mistral_api_key):
    """Single explanation quality evaluation by Mistral."""
    client     = Mistral(api_key=mistral_api_key.strip())
    events_str = '\n'.join([
        f"  {e['rank']}. [{e['template']}]"
        for e in top_events])
    prompt = f"""Expert cybersecurity analyst evaluation.
Predicted: {predicted}
Top events: {events_str}
Explanation: {explanation}
Rate (1-5) usefulness and readability.
JSON only: {{"usefulness": X, "readability": X,
             "comment": "..."}}"""
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=150)
    text = response.choices[0].message.content.strip()
    text = text.replace('```json','').replace('```','').strip()
    return json.loads(text)
