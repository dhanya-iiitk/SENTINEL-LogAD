"""
SENTINEL Explainability Pipeline

Modules:
    ig_attribution : Integrated Gradients event attribution
    groq_explain   : Groq LLM cyber-threat explanation
    evaluate       : Mistral AI quality evaluation +
                     grounding/consistency validation
"""
from .ig_attribution import integrated_gradients, get_top_events
from .groq_explain   import groq_explain, build_prompt
from .evaluate       import (mistral_evaluate, grounding_score,
                              consistency_check)
