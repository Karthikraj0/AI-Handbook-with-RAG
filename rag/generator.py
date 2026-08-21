import ollama


LLM_MODEL = "gpt-oss:latest"


def _build_prompt(query, context):
    return f"""
You are a company policy assistant.

Answer the employee's question using ONLY the information
provided in the policy context below.

If the answer cannot be found in the policy context,
say:

"I couldn't find that information in the company policies."

Do not make up information.
Do not use outside knowledge.
Do not assume or invent company policies.

POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{query}

ANSWER:
"""


def generate_answer(query, context):

    prompt = _build_prompt(query, context)

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

    return response["message"]["content"]


def generate_answer_stream(query, context):

    prompt = _build_prompt(query, context)

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

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content