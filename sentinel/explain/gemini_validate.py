from google import genai
import json

THREAT_WORDS = ['attack', 'threat', 'malicious', 'exploit',
                'breach', 'intrusion', 'compromise', 'adversar']


def gemini_score(explanation, top_events, predicted,
                 gemini_api_key):
    client     = genai.Client(api_key=gemini_api_key)
    events_str = '\n'.join([
        f"  {e['rank']}. [{e['template']}]"
        for e in top_events])
    prompt = f"""You are an expert cybersecurity analyst.
Predicted label: {predicted}
Top log events: {events_str}
Explanation: {explanation}
Rate (1-5): usefulness and readability.
JSON only: {{"usefulness": X, "readability": X, "comment": "..."}}"""
    response = client.models.generate_content(
        model='gemini-1.5-flash', contents=prompt)
    text = response.text.strip()
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
