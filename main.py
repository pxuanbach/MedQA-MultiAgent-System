"""V0 / V3 entry point for the MedQA Multi-Agent System.

Usage (programmatic)
--------------------
    # V0 — direct LLM (no RAG, no memory)
    from main import agent
    agent.invoke("A 45-year-old patient presents with...")

    # V3 — full pipeline (rewrite → retrieve → answer + memory)
    from medqa_multi_agents import invoke
    invoke("A 45-year-old patient presents with...")

Usage (CLI)
-----------
    python main.py "A 45-year-old patient..."
    python main.py --no-memory "A 45-year-old patient..."

Environment variables
--------------------
LM_STUDIO_URL    : LM Studio base URL  (default http://localhost:1234/v1)
LM_STUDIO_MODEL  : model name          (default qwen/qwen3-8b)
ENABLE_MEMORY    : enable V3 memory     (default true)
"""

from __future__ import annotations

import os
import sys

from medqa_multi_agents import _get_model, invoke
from medqa_multi_agents.agents.answerer import Answer, _load_answerer_prompt

# ---------------------------------------------------------------------------
# Singleton model (one per process)
# ---------------------------------------------------------------------------
_model = None


def _get_singleton_model():
    global _model
    if _model is None:
        _model = _get_model()
    return _model


# ---------------------------------------------------------------------------
# V0 agent — direct LLM, no RAG, no memory
# ---------------------------------------------------------------------------


class _Agent:
    """Callable agent that exposes ``.invoke(prompt)``."""

    def __init__(self, name: str):
        self._name = name

    def invoke(self, prompt: str) -> str:
        """Answer a MedQA question with direct LLM.

        V0: no RAG, no memory. Model is called directly with the answerer
        system prompt and the question as user input.
        """
        model = _get_singleton_model()
        system_prompt = _load_answerer_prompt()
        response = model.with_structured_output(Answer).invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ])
        return response.answer

    def __repr__(self) -> str:
        return f"<{self._name} agent>"


agent = _Agent("V0")





# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python main.py [flags] '<question>'")
#         print()
#         print("Flags")
#         print("  --no-memory    Disable V3 long/short-term memory (default: memory enabled)")
#         print()
#         print("Modes")
#         print("  V0  : direct LLM, no RAG, no memory")
#         print("  V3  : full pipeline + memory (default when --rag omitted)")
#         sys.exit(1)

#     # Parse flags
#     use_memory = "--no-memory" not in sys.argv

#     # Collect positional args (the question)
#     parts = [a for a in sys.argv[1:] if a != "--no-memory"]
#     question = " ".join(parts)

#     if not question.strip():
#         print("Error: question cannot be empty")
#         sys.exit(1)

#     # V0: direct LLM (always, no RAG)
#     # V3: full workflow via invoke() with memory controlled by ENABLE_MEMORY
#     prev = os.environ.get("ENABLE_MEMORY")
#     os.environ["ENABLE_MEMORY"] = "true" if use_memory else "false"
#     try:
#         answer = invoke(question)
#     finally:
#         if prev is not None:
#             os.environ["ENABLE_MEMORY"] = prev
#         else:
#             os.environ.pop("ENABLE_MEMORY", None)

#     print(answer)

# answer = invoke("""
# A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient. The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given?

# options: {"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"}
# """)

# answer = invoke("""
# A 5-year-old girl is brought to the emergency department by her mother because of multiple episodes of nausea and vomiting that last about 2 hours. During this period, she has had 6–8 episodes of bilious vomiting and abdominal pain. The vomiting was preceded by fatigue. The girl feels well between these episodes. She has missed several days of school and has been hospitalized 2 times during the past 6 months for dehydration due to similar episodes of vomiting and nausea. The patient has lived with her mother since her parents divorced 8 months ago. Her immunizations are up-to-date. She is at the 60th percentile for height and 30th percentile for weight. She appears emaciated. Her temperature is 36.8°C (98.8°F), pulse is 99/min, and blood pressure is 82/52 mm Hg. Examination shows dry mucous membranes. The lungs are clear to auscultation. Abdominal examination shows a soft abdomen with mild diffuse tenderness with no guarding or rebound. The remainder of the physical examination shows no abnormalities. Which of the following is the most likely diagnosis?

# options: {"A": "Cyclic vomiting syndrome", "B": "Gastroenteritis", "C": "Hypertrophic pyloric stenosis", "D": "Gastroesophageal reflux disease"}
# """)

answer = invoke("""
A 40-year-old woman presents with difficulty falling asleep, diminished appetite, and tiredness for the past 6 weeks. She says that, despite going to bed early at night, she is unable to fall asleep. She denies feeling anxious or having disturbing thoughts while in bed. Even when she manages to fall asleep, she wakes up early in the morning and is unable to fall back asleep. She says she has grown increasingly irritable and feels increasingly hopeless, and her concentration and interest at work have diminished. The patient denies thoughts of suicide or death. Because of her diminished appetite, she has lost 4 kg (8.8 lb) in the last few weeks and has started drinking a glass of wine every night instead of eating dinner. She has no significant past medical history and is not on any medications. Which of the following is the best course of treatment in this patient?

options: {"A": "Diazepam", "B": "Paroxetine", "C": "Zolpidem", "D": "Trazodone"}
""")

print("===" * 10, "\n")
print(answer)