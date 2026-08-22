import re
import ollama


LLM_MODEL = "llama3.2:3b"


def _build_prompt(query, context, is_custom_doc=False):
    if is_custom_doc:
        return f"""
You are an AI Document Analysis Assistant.

Answer the user's question using ONLY the provided DOCUMENT CONTEXT below.

Instructions:
1. Level of Detail & User Constraints:
   - Provide a thorough, well-explained, and informative answer based on the document.
   - If the user specifies a particular length, format, or detail level (e.g., "brief summary", "in 2 sentences", "detailed breakdown", "give 3 bullet points", "explain in detail", "high-level overview"), STRICTLY adapt your answer length and format to match the user's request.
   - Otherwise, provide a detailed, well-structured answer explaining the key clauses, conditions, and facts found in the context.
2. Formatting & Lists:
   - When presenting multiple points or lists, ALWAYS place every point on its own NEW line with a markdown hyphen and blank line separation (e.g., "- Point 1\\n\\n- Point 2").
   - NEVER concatenate or join multiple bullet points onto the same line.
3. Tone & Directness:
   - Start directly with the answer. Do NOT use conversational preambles or intros like "Here is the answer...", "Based on the document provided...", or "To answer your question...".
   - Do NOT include document metadata headers unless explicitly requested.
   - Do NOT add trailing filler like "Would you like to know more?", "Feel free to ask!", or closing signatures.
4. Grounding & Fallback:
   - Answer strictly from the DOCUMENT CONTEXT. Do not invent facts or use outside knowledge.
   - If the answer cannot be found in the document context, say:
     "I couldn't find that information in the uploaded document."

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""
    else:
        return f"""
You are a helpful AI Assistant for company policies.

Answer the employee's question using ONLY the provided POLICY CONTEXT.

Instructions:
1. Level of Detail & User Constraints:
   - Provide a clear, natural, and informative answer with the relevant facts.
   - If the user specifies a particular length, format, or detail level (e.g., "brief summary", "in 2 bullet points", "detailed explanation"), STRICTLY follow the user's requested format and length.
   - Otherwise, provide a clear, well-structured answer covering the relevant policy facts.
2. Formatting & Lists:
   - When presenting multiple points or lists, ALWAYS place every point on its own NEW line with a markdown hyphen (e.g., "- Point 1\\n\\n- Point 2").
   - NEVER combine multiple bullet points onto the same line.
3. Tone & Directness:
   - Start directly with the answer. Do NOT use conversational preambles or intros like "Here is the answer...", "Here's a clear and helpful answer...", "Based on the policy context...", or "To answer your question...".
   - Do NOT include document metadata headers like Policy IDs, version numbers, or effective dates unless specifically asked.
   - Do NOT add trailing filler like "Would you like to know more?", "Feel free to ask!", or closing signatures.
   - Do NOT answer math, trivia, coding, or off-topic questions.
4. Fallback:
   - If the answer cannot be found in the policy context, say:
     "I couldn't find that information in the company policies."

POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{query}

ANSWER:
"""


def _clean_response(content: str) -> str:
    """Strips any thinking/reasoning blocks and formats bullet points onto separate lines."""
    if not content:
        return ""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    elif "<think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # Format inline unicode bullet points (e.g., "text. • Next point") into proper markdown line breaks
    content = re.sub(r'([^\n])\s*[•*]\s+', r'\1\n\n- ', content)
    # Format line-starting unicode bullets into standard markdown hyphens
    content = re.sub(r'^\s*[•*]\s+', '- ', content, flags=re.MULTILINE)

    return content.strip()


def generate_answer(query, context, is_custom_doc=False):

    prompt = _build_prompt(query, context, is_custom_doc=is_custom_doc)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.2
        },
        keep_alive="1h"
    )

    raw_content = response.get("message", {}).get("content", "")
    return _clean_response(raw_content)


def generate_answer_stream(query, context, is_custom_doc=False):

    prompt = _build_prompt(query, context, is_custom_doc=is_custom_doc)

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.2
        },
        stream=True,
        keep_alive="1h"
    )

    buffer = ""
    has_seen_think_end = False

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if not content:
            continue

        if "<think>" in content or "</think>" in content:
            buffer += content
            if "</think>" in buffer:
                has_seen_think_end = True
                after_think = buffer.split("</think>", 1)[1]
                buffer = ""
                clean_start = after_think.lstrip("\r\n ")
                if clean_start:
                    yield clean_start
            continue

        # Ensure unicode bullet points start on a fresh newline in the live UI stream
        if "•" in content:
            content = content.replace("•", "\n\n- ")

        yield content

    if buffer and not has_seen_think_end:
        yield _clean_response(buffer)