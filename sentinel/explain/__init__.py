"""
SENTINEL Explainability Pipeline

Modules:
    ig_attribution : Integrated Gradients event attribution
    groq_explain   : Multi-LLM cyber-threat explanation
                     (GPT-OSS-120B, LLaMA-4-Scout, Qwen3-32B)
    evaluate       : Cross-LLM similarity evaluation (Mistral AI)
                     + grounding/consistency validation
"""
from .ig_attribution import integrated_gradients, get_top_events
from .groq_explain   import (get_three_explanations,
                              groq_explain, build_prompt)
from .evaluate       import (mistral_evaluate_similarity,
                              mistral_evaluate,
                              grounding_score,
                              consistency_check)
