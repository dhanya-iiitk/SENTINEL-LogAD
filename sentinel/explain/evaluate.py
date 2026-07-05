"""
Explainability evaluation module for SENTINEL.
Uses Mistral AI (mistral-small-latest) as independent
evaluator of Groq-generated explanations.
"""
from mistralai import Mistral
import json

THREAT_WORDS = ['attack', 'threat', 'malicious', 'exploit',
                'breach', 'intrusion', 'compromise', 'adversar']


def mistral_evaluate(explanation, top_events, predicted,
                     mistral_api_key):
    client     = Mistral(api_key=mistral_api_key)
    events_str = '\n'.join([
        f"  {e['rank']}. [{e['template']}]"
        for e in top_events])
    prompt = f"""You are an expert cybersecurity analyst
evaluating the quality of a log anomaly explanation.
Predicted label: {predicted}
Top contributing log events (by Integrated Gradients):
{events_str}
Generated explanation:
{explanation}
Rate (1-5):
1. Usefulness: specific, plausible, references events, actionable?
2. Readability: clear, concise, no hallucinations?
JSON only: {{"usefulness": X, "readability": X, "comment": "..."}}"""
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=150)
    text = response.choices[0].message.content.strip()
    text = text.replace('```json','').replace('```','').strip()
    return json.loads(text)


def grounding_score(explanation, top_events):
    grounded = 0
    for e in top_events:
        terms = [w for w in e['template'].split()
                 if len(w) > 3 and (w.isupper() or w.isalpha())]
        if any(t.lower() in explanation.lower() for t in terms):
            grounded += 1
    return grounded / len(top_events) if top_events else 0.0


def consistency_check(explanation, predicted):
    claims_threat = any(w in explanation.lower()
                        for w in THREAT_WORDS)
    if predicted == 'Normal':
        return not claims_threat
    return True
