import sys
import time
import io

# Ensure UTF-8 stdout for Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from rag.query_rewriter import reformulate_query
from rag.retrieve import retrieve_documents
from rag.generator import generate_answer
from rag.pipeline import ask_question, ask_question_stream

print("--- WARMING UP OLLAMA MODELS (Qwen3 & GPT-OSS) ---")
ask_question("wfh")

test_cases = [
    ("Relevant: WFH rules", "How many days per week can employees work from home?"),
    ("Relevant: Edge acronym", "wfh?"),
    ("Relevant: Health Insurance", "What health insurance benefits does the company provide?"),
    ("Relevant: Sick Leave", "How many days of sick leave am I entitled to?"),
    ("Irrelevant: Rejection", "What is the speed of light in vacuum?"),
    ("Irrelevant: Out-of-Scope", "How do I cook pasta carbonara?")
]

records = []

for label, q in test_cases:
    t0 = time.perf_counter()
    ref_q = reformulate_query(q)
    t1 = time.perf_counter()
    
    results = retrieve_documents(q, n_results=5, reformulated_query=ref_q)
    t2 = time.perf_counter()
    
    top_dist = results['distances'][0][0] if results and results.get('distances') and results['distances'][0] else 1.0
    is_blocked = top_dist > 0.47
    
    if not is_blocked:
        context = '\n\n'.join(results['documents'][0])
        ans = generate_answer(q, context)
    else:
        ans = "I could not find that information..."
    t3 = time.perf_counter()
    
    t_rewrite = t1 - t0
    t_retrieval = t2 - t1
    t_gen = t3 - t2
    t_total = t3 - t0
    records.append((label, t_rewrite, t_retrieval, t_gen, t_total, is_blocked, ref_q))

print("\n" + "="*95)
print(f"{'QUERY CASE':<30} | {'REWRITE':<10} | {'RETRIEVAL':<10} | {'GENERATION':<10} | {'TOTAL':<10}")
print("="*95)
for r in records:
    gen_str = f"{r[3]:.3f}s" if not r[5] else "0.000s (skip)"
    print(f"{r[0]:<30} | {r[1]:.3f}s     | {r[2]:.3f}s     | {gen_str:<10} | {r[4]:.3f}s")
print("="*95)

print("\n--- REFORMULATION ACCURACY CHECK ---")
for r in records:
    print(f"[{r[0]}] -> Reformulated: \"{r[6]}\"")

print("\n--- STREAMING LATENCY & TIME TO FIRST TOKEN (TTFT) ---")
stream_queries = [
    "How many days per week can employees work from home?",
    "What is the procedure for applying for sick leave?"
]
for sq in stream_queries:
    t_start = time.perf_counter()
    res = ask_question_stream(sq)
    first_token_t = None
    token_count = 0
    for chunk in res['stream']:
        if first_token_t is None:
            first_token_t = time.perf_counter()
        token_count += 1
    t_end = time.perf_counter()
    ttft = first_token_t - t_start if first_token_t else 0
    total_t = t_end - t_start
    print(f"Query: \"{sq}\"")
    print(f"  -> Time to First Token (TTFT): {ttft:.3f} s (User starts reading)")
    print(f"  -> Total Streaming Duration  : {total_t:.3f} s ({token_count} chunks)")
