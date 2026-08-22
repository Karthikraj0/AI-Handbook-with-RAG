import sys
import os

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag.pipeline import ask_question, ask_question_stream
from rag.query_rewriter import reformulate_query
from rag.vectorstore import get_collection
from rag.embeddings import create_embedding

TEST_QUERIES = [
    "wfh?",
    "wfh",
    "WFH",
    "wfh policy",
    "wfh polciy",
    "Can I wfh?",
    "Can I work remotely?",
    "work from home",
    "pto?",
    "pto policy",
    "how many leave i get?",
    "travell allowance?"
]

def run_tests():
    print("=" * 100)
    print("TESTING LLM-BASED QUERY REFORMULATION SUITE")
    print("=" * 100)
    
    collection = get_collection()

    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n--- [Test {idx}/{len(TEST_QUERIES)}] Query: '{query}' ---")
        
        # 1. Reformulation
        reformulated = reformulate_query(query)
        print(f"1. Original Query       : {query}")
        print(f"2. Reformulated Query   : {reformulated}")
        
        # 2. Vector Search Metrics
        emb = create_embedding(reformulated)
        res = collection.query(query_embeddings=[emb], n_results=3)
        
        distances = res["distances"][0] if res and res.get("distances") and res["distances"][0] else [1.0]
        top_dist = distances[0]
        sources = res["metadatas"][0] if res and res.get("metadatas") and res["metadatas"][0] else []
        top_source = f"{sources[0].get('source', 'Unknown')} (Page {sources[0].get('page', '?')})" if sources else "None"
        
        passes_threshold = top_dist <= 0.47
        status_str = "PASSED (<= 0.47)" if passes_threshold else "BLOCKED (> 0.47)"
        
        print(f"3. Top Retrieved Distance: {top_dist:.4f}")
        print(f"4. Top Source Match      : {top_source}")
        print(f"5. Threshold Status      : {status_str}")
        
        # 3. Pipeline End-to-End Answer
        result = ask_question(query)
        answer = result["answer"].strip()
        first_line_answer = answer.split("\n")[0][:140]
        print(f"6. Generated Answer      : {first_line_answer}...")

    # Test Streaming
    print("\n" + "=" * 100)
    print("TESTING STREAMING FUNCTIONALITY (ask_question_stream)")
    print("=" * 100)
    stream_res = ask_question_stream("wfh?")
    stream = stream_res["stream"]
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    streamed_text = "".join(chunks).strip()
    print(f"Streamed Chunks Count : {len(chunks)}")
    print(f"Streamed Output Sample: {streamed_text.splitlines()[0][:140]}...")
    print("\n" + "=" * 100)
    print("ALL 12 TESTS COMPLETED SUCCESSFULLY")
    print("=" * 100)

if __name__ == "__main__":
    run_tests()
