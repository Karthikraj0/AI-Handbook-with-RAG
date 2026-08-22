import logging
import ollama

logger = logging.getLogger(__name__)

REWRITER_MODEL = "gpt-oss:latest"

REFORMULATION_SYSTEM_PROMPT = """You are a query reformulation component for a company-policy RAG system.

Rewrite the user's query into a clear, concise search query suitable for semantic retrieval from company policy documents.

Rules:
1. Preserve the user's original intent.
2. Correct obvious spelling and grammar mistakes.
3. Expand common workplace abbreviations and acronyms when appropriate:
   - WFH / wfh -> work from home
   - PTO / pto -> paid time off / annual leave
   - LOP / lop -> loss of pay / unpaid leave
   - TA/DA -> travel allowance / daily per diem allowance
   - PF -> provident fund
   - VPN -> virtual private network remote access
   - BYOD -> bring your own device
   - NDA -> non-disclosure agreement
   - HR -> human resources
4. Clarify informal wording, slang, or single-word queries into searchable policy terms.
5. Do NOT answer the question.
6. Do NOT invent company policies, rules, numbers, dates, or facts.
7. Do NOT add information that is not implied by the user's query.
8. Return ONLY the rewritten search query. Do NOT add quotes, markdown formatting, prefixes, or explanations.
9. Keep the rewritten query concise.

Examples:

User: "wfh?"
Output: work from home policy

User: "Can I wfh?"
Output: Can I work from home?

User: "wfh polciy"
Output: work from home policy

User: "how many leave i get?"
Output: employee leave entitlement

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
    Reformulates the user's query into a clear search query for semantic vector retrieval.
    If the LLM call fails, times out, or produces an invalid output, safely falls back to the original query.
    """
    if not query or not query.strip():
        return query

    clean_query = query.strip()

    try:
        response = ollama.chat(
            model=REWRITER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": REFORMULATION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"User query:\n{clean_query}"
                }
            ],
            options={
                "temperature": 0.0,
                "top_p": 0.9,
            },
            keep_alive="1h"
        )

        reformulated = response.get("message", {}).get("content", "").strip()

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
