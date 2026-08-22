import sys

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag.pipeline import ask_question, DISTANCE_THRESHOLD
from rag.query_rewriter import reformulate_query
from rag.retrieve import retrieve_documents

RELEVANT_EDGE_CASES = [
    # Acronyms & Short forms
    ("wfh?", "Work From Home Policy"),
    ("WFH", "Work From Home Policy"),
    ("wfh policy", "Work From Home Policy"),
    ("wfh polciy", "Work From Home Policy (Typo)"),
    ("Can I wfh?", "Work From Home Question"),
    ("pto?", "Paid Time Off (Leave Policy)"),
    ("pto balance", "Paid Time Off (Leave Policy)"),
    ("lop leave rules", "Loss of Pay / Unpaid Leave"),
    ("vpn connection guide", "VPN / Remote Access"),
    ("pf contribution percentage", "Provident Fund / Benefits"),
    ("ta/da reimbursement rates", "Travel Allowance / Per Diem"),
    ("byod security guidelines", "IT Security / Devices"),
    ("hr email contact", "General Handbook / HR"),
    ("nda policy requirements", "IT Security / NDA"),

    # Short / Single-word queries
    ("leave?", "Leave Policy"),
    ("sick leave", "Sick Leave Policy"),
    ("maternity", "Maternity Leave Policy"),
    ("paternity", "Paternity Leave Policy"),
    ("resignation", "Notice Period / Resignation"),
    ("per diem allowance", "Travel Policy / Per Diem"),
    ("gym subsidy", "Wellness & Gym Allowance"),
    ("laptop policy", "IT Security / Hardware"),

    # Conversational Typos
    ("can i wrk from hmoe?", "WFH with Typos"),
    ("sick leav apply procedure", "Sick Leave with Typo"),
    ("reimbursment for travel", "Reimbursement with Typo"),
    ("can i take unpaid leve?", "Unpaid Leave with Typo"),
]

IRRELEVANT_EDGE_CASES = [
    ("ROI of NVDA stock", "Finance / External Stock"),
    ("AWS EC2 pricing model", "Cloud Infrastructure"),
    ("NASA JWST orbital path", "Space Science"),
    ("NATO member countries", "Geopolitics"),
    ("pizza recipe", "Cooking / Food"),
    ("how to change car tire", "General Maintenance"),
]


def test_edge_cases():
    print("=" * 100)
    print("TESTING RELEVANT EDGE CASES: QUERY REFORMULATION + RETRIEVAL + ANSWER GENERATION")
    print(f"Active Guardrail Threshold: {DISTANCE_THRESHOLD}")
    print("=" * 100)

    print(f"\n{'#':<3} | {'QUERY':<28} | {'REFORMULATED':<34} | {'DIST':<6} | {'STATUS':<7} | {'SOURCE'}")
    print("-" * 100)

    passed_count = 0
    total_rel = len(RELEVANT_EDGE_CASES)

    results_details = []

    for idx, (query, label) in enumerate(RELEVANT_EDGE_CASES, 1):
        reformulated = reformulate_query(query)
        res = retrieve_documents(query, n_results=1, reformulated_query=reformulated)
        
        if res and res.get("distances") and res["distances"][0]:
            dist = res["distances"][0][0]
            source = res["metadatas"][0][0].get("source", "Unknown")
            page = res["metadatas"][0][0].get("page", "?")
            passed = dist <= DISTANCE_THRESHOLD
        else:
            dist = 1.0
            source = "None"
            page = "?"
            passed = False

        status_icon = "PASS" if passed else "FAIL"
        if passed:
            passed_count += 1

        print(f"{idx:<3} | {query:<28} | {reformulated[:33]:<34} | {dist:<6.4f} | {status_icon:<7} | {source} (p.{page})", flush=True)

        # Get generated answer
        rag_res = ask_question(query)
        results_details.append({
            "idx": idx,
            "query": query,
            "label": label,
            "reformulated": reformulated,
            "dist": dist,
            "source": f"{source} (Page {page})",
            "passed": passed,
            "answer": rag_res["answer"].strip()
        })

    print("-" * 100)
    print(f"Relevant Edge Cases Pass Rate: {passed_count}/{total_rel} ({passed_count/total_rel*100:.1f}%)")

    # Sample detailed answers
    print("\n" + "=" * 100)
    print("DETAILED GENERATED ANSWERS SAMPLE FOR KEY EDGE CASES:")
    print("=" * 100)
    sample_indices = [1, 2, 4, 6, 8, 11, 15, 17, 23]
    for d in results_details:
        if d["idx"] in sample_indices:
            print(f"\n--- [Edge Case #{d['idx']}] \"{d['query']}\" ({d['label']}) ---")
            print(f"Reformulated -> \"{d['reformulated']}\" (Dist: {d['dist']:.4f}, Source: {d['source']})")
            print(f"Answer:\n{d['answer']}")
            print("-" * 60)

    # Irrelevant Edge Cases
    print("\n" + "=" * 100)
    print("TESTING IRRELEVANT / OUT-OF-DOMAIN EDGE CASES (Should be Rejected > 0.47)")
    print("=" * 100)
    irr_blocked = 0
    for idx, (query, label) in enumerate(IRRELEVANT_EDGE_CASES, 1):
        reformulated = reformulate_query(query)
        rag_res = ask_question(query)
        answer = rag_res["answer"]
        blocked = "couldn't find that information" in answer.lower()
        if blocked:
            irr_blocked += 1
        print(f"#{idx} | Query: '{query}' -> Reformulated: '{reformulated}'")
        print(f"   Answer: {answer[:90]}... -> {'BLOCKED (Correct)' if blocked else 'LEAKED (False Positive)'}")

    print("-" * 100)
    print(f"Irrelevant Edge Cases Block Rate: {irr_blocked}/{len(IRRELEVANT_EDGE_CASES)} ({irr_blocked/len(IRRELEVANT_EDGE_CASES)*100:.1f}%)")
    print("=" * 100)


if __name__ == "__main__":
    test_edge_cases()
