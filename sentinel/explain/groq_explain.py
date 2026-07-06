"""
SENTINEL Explainability — Multi-LLM Generator
Generates cyber-threat explanations using 3 independent LLMs:
  LLM1: GPT-OSS-120B      (Groq)
  LLM2: LLaMA-4-Scout-17B (Groq)
  LLM3: Qwen3-32B         (Groq)
"""
from groq import Groq


def build_prompt(top_events, anomaly_score, threshold,
                 predicted, dataset_name='BGL'):
    events_str = '\n'.join([
        f"  {e['rank']}. [{e['template']}] "
        f"(attribution: {e['attribution']:.3f})"
        for e in top_events])
    if predicted == 'Anomalous':
        return f"""You are an expert cybersecurity analyst.
A log session from {dataset_name} was flagged as ANOMALOUS.
Anomaly Score : {anomaly_score:.4f} (threshold={threshold:.4f})
Top contributing log events (by Integrated Gradients):
{events_str}
1. Explain WHY this session is anomalous (2-3 sentences).
2. Identify the most likely fault or cyber threat.
3. Suggest one immediate remediation step.
Be concise and technical."""
    else:
        return f"""You are an expert cybersecurity analyst.
A log session from {dataset_name} was classified as NORMAL.
Anomaly Score : {anomaly_score:.4f} (threshold={threshold:.4f})
Top characteristic log events:
{events_str}
Confirm in 1-2 sentences why this reflects normal behavior.
Do NOT describe it as anomalous."""


def get_three_explanations(top_events, anomaly_score,
                            threshold, predicted,
                            groq_api_key,
                            dataset_name='BGL'):
    """
    Generate explanations from 3 independent LLMs
    using the same IG-grounded prompt.
    Returns dict of {llm_name: explanation_text}
    """
    prompt  = build_prompt(top_events, anomaly_score,
                            threshold, predicted, dataset_name)
    client  = Groq(api_key=groq_api_key)
    results = {}

    models = {
        'LLM1_GPT120B':  'openai/gpt-oss-120b',
        'LLM2_LLaMA4':   'meta-llama/llama-4-scout-17b-16e-instruct',
        'LLM3_Qwen3':    'qwen/qwen3-32b',
    }

    for name, model_id in models.items():
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=300)
            results[name] = r.choices[0].message.content.strip()
        except Exception as e:
            results[name] = f"Error: {e}"

    return results, prompt


def groq_explain(top_events, anomaly_score, threshold,
                  predicted, groq_api_key,
                  dataset_name='BGL',
                  model_name='openai/gpt-oss-120b'):
    """Single LLM explanation (primary generator)."""
    client  = Groq(api_key=groq_api_key)
    prompt  = build_prompt(top_events, anomaly_score,
                            threshold, predicted, dataset_name)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=300)
    return response.choices[0].message.content.strip()
