import logging
import re
import ollama

logger = logging.getLogger(__name__)

REWRITER_MODEL = "qwen3:1.7b"

REFORMULATION_SYSTEM_PROMPT = """You are a query reformulation component for a company-policy RAG system.

Rewrite the user's query into a clear, concise search query suitable for semantic retrieval from company policy documents.

Rules:
1. Preserve the user's original intent.
2. Correct obvious spelling and grammar mistakes.
3. Expand common workplace abbreviations and acronyms:
   - WFH / wfh -> work from home policy
   - PTO / pto -> paid time off policy
   - LOP / lop -> loss of pay leave policy
   - TA/DA -> travel allowance and daily per diem allowance
   - PF -> provident fund
   - VPN -> virtual private network remote access
   - BYOD -> bring your own device security guidelines
   - NDA -> non-disclosure agreement
   - HR -> human resources
4. Clarify informal wording, slang, typos, or single-word queries into searchable policy terms.
5. Do NOT answer the question.
6. Do NOT invent company policies, rules, numbers, dates, or facts.
7. Do NOT add information that is not implied by the user's query.
8. Return ONLY the rewritten search query. Do NOT add quotes, markdown formatting, prefixes, or explanations.
9. Keep the rewritten query concise.
10. Strip out any non-work chatter, math problems (e.g., 'and 4+5'), or trivia attached to a policy query.

Examples:

User: "wfh?"
Output: work from home policy

User: "WFH policy"
Output: work from home policy

User: "how many days can i work from home"
Output: how many days can employees work from home

User: "health insurence benefits"
Output: health insurance benefits

User: "Can I wfh?"
Output: Can I work from home?

User: "wfh polciy"
Output: work from home policy

User: "how many leave i get?"
Output: employee leave entitlement

User: "pto?"
Output: paid time off policy

User: "pto policy"
Output: paid time off policy

User: "can i work remotely?"
Output: Can I work remotely?

User: "What is the reimbursement process for travel?"
Output: travel reimbursement process

User: "maternity"
Output: maternity leave policy and eligibility

User: "wifi password"
Output: office wifi password and guest network access

User: "can i wrk from hmoe?"
Output: Can I work from home?
"""


def reformulate_query(query: str) -> str:
    """
    Reformulates the user's query into a clear search query for semantic vector retrieval using Qwen3 (1.7B).
    Configured with non-thinking mode for fast inference.
    If the LLM call fails, times out, or produces an invalid output, safely falls back to the original query.
    """
    if not query or not query.strip():
        return query

    clean_query = query.strip()

    try:
        kwargs = {
            "model": REWRITER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": REFORMULATION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"User: {clean_query}\nOutput:"
                }
            ],
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
            },
            "keep_alive": "1h"
        }

        # Attempt to pass think=False if supported by current Ollama version
        try:
            response = ollama.chat(**kwargs, think=False)
        except TypeError:
            response = ollama.chat(**kwargs)

        reformulated = response.get("message", {}).get("content", "").strip()

        # Strip any <think>...</think> reasoning blocks if present
        if "<think>" in reformulated and "</think>" in reformulated:
            reformulated = re.sub(r"<think>.*?</think>", "", reformulated, flags=re.DOTALL).strip()

        # Remove surrounding quotes or backticks if generated
        if (reformulated.startswith('"') and reformulated.endswith('"')) or \
           (reformulated.startswith("'") and reformulated.endswith("'")) or \
           (reformulated.startswith("`") and reformulated.endswith("`")):
            reformulated = reformulated[1:-1].strip()

        # Remove markdown bold/italic wrappers if any
        if reformulated.startswith("**") and reformulated.endswith("**") and len(reformulated) > 4:
            reformulated = reformulated[2:-2].strip()

        # Strip prefixes like "Output:", "Search query:", "Rewritten query:"
        prefixes_to_strip = [
            "output:", "search query:", "rewritten query:", "reformulated query:", "result:"
        ]
        for prefix in prefixes_to_strip:
            if reformulated.lower().startswith(prefix):
                reformulated = reformulated[len(prefix):].strip()

        # Safety validation:
        # If output is empty or contains multi-paragraph answers (attempted answering), fall back to original query
        if not reformulated:
            logger.warning("[QueryRewriter] Empty reformulation received. Falling back to original query.")
            return clean_query

        if len(reformulated.split("\n")) > 3 or len(reformulated) > 250:
            logger.warning("[QueryRewriter] Output too lengthy or conversational. Falling back to original query.")
            return clean_query

        logger.info(f"[QueryRewriter] Original: '{clean_query}' -> Reformulated: '{reformulated}'")
        return reformulated

    except Exception as e:
        logger.warning(f"[QueryRewriter] LLM reformulation encountered error ({e}). Falling back to original query.")
        return clean_query
