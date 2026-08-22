import sys
import numpy as np

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag.vectorstore import get_collection
from rag.embeddings import create_embedding
from rag.query_rewriter import reformulate_query

THRESHOLD = 0.47

RELEVANT_QUESTIONS_50 = [
    # Leave Policy (1-7)
    "How many annual leave days do employees get each year?",
    "What is the policy for taking sick leave?",
    "Can unused annual leave be carried forward to the next year?",
    "What is the duration of maternity leave?",
    "Are fathers eligible for paternity leave?",
    "How do I request an unpaid leave of absence?",
    "What documents are required for bereavement leave?",

    # Attendance Policy (8-14)
    "What are the standard office operating hours?",
    "What is the grace period for arriving late to work?",
    "How is employee attendance tracked daily?",
    "What disciplinary action is taken for unauthorized absences?",
    "Can employees take flexible working hours?",
    "What should I do if I am running late for work?",
    "How does the company handle half-day attendance?",

    # IT Security Policy (15-21)
    "What is the required password complexity for company accounts?",
    "How often do I need to change my login password?",
    "Is it allowed to use personal USB drives on company laptops?",
    "What should I do if I receive a suspicious phishing email?",
    "Can I install third-party software on my work laptop without permission?",
    "What is the policy regarding remote access and VPN usage?",
    "How should sensitive customer data be stored and encrypted?",

    # Employee Benefits (22-28)
    "What medical insurance coverage does the company provide?",
    "Are dependents or family members covered under health insurance?",
    "Does the company offer a provident fund or retirement scheme?",
    "What wellness and gym allowance programs are available?",
    "Is there a learning and development budget for employees?",
    "Does the company provide life insurance coverage?",
    "What perks or discounts are available to full-time employees?",

    # Reimbursement Policy (29-35)
    "What is the deadline for submitting monthly expense reimbursement claims?",
    "Are internet and mobile bills eligible for reimbursement?",
    "What receipts and invoices are mandatory for expense approval?",
    "What is the maximum reimbursement amount for client entertainment?",
    "How are team lunch expenses claimed and approved?",
    "Can I get reimbursed for professional certifications and courses?",
    "How long does it take for finance to process reimbursement claims?",

    # Travel Policy (36-42)
    "What is the daily allowance (per diem) for domestic business travel?",
    "What class of flight is allowed for international business trips?",
    "How do I book company hotels and transport for business trips?",
    "Are meals during business travel reimbursed by the company?",
    "Can I use rideshare or taxi services during official travel?",
    "What is the policy on extending business trips for personal travel?",
    "Who needs to approve business travel before booking tickets?",

    # Work From Home (WFH) Policy (43-48)
    "How many days per week am I allowed to work from home?",
    "Does the company provide an ergonomic chair or monitor for home office?",
    "Are all employees eligible for remote work?",
    "What are the core communication hours while working remotely?",
    "Can I work remotely from another country or state?",
    "What internet speed is required for home office setup?",

    # General Employee Handbook & Conduct (49-50)
    "What is the company code of conduct regarding workplace harassment?",
    "What is the notice period required when resigning from the company?"
]

IRRELEVANT_QUESTIONS_50 = [
    # General Science & Space (1-6)
    "What is the average distance from the Earth to the Sun?",
    "How do black holes form in outer space?",
    "What is the atomic number of Gold on the periodic table?",
    "What is Newton's third law of motion?",
    "Why does the human body require vitamin D?",
    "What is the process of photosynthesis in green plants?",

    # Cooking & Food (7-12)
    "How do you bake a classic Italian pizza dough from scratch?",
    "What are the key ingredients for authentic Mexican guacamole?",
    "How long should you boil an egg for a soft yolk?",
    "What is the secret to making a fluffy French omelet?",
    "How do you brew a traditional Japanese matcha tea?",
    "What spices are used in Indian garam masala?",

    # Programming & Tech Trivia (13-18)
    "How do I implement a Red-Black tree in C++?",
    "What is the difference between TCP and UDP protocols?",
    "How do you write a docker-compose file for PostgreSQL?",
    "What are React hooks and how does useEffect work?",
    "How does the QuickSort algorithm achieve O(n log n) time complexity?",
    "What is the difference between asynchronous and multithreaded execution?",

    # World Geography & Travel (19-24)
    "What is the capital city of Australia?",
    "Which is the longest river in South America?",
    "What are the top tourist attractions to visit in Kyoto?",
    "How high is Mount Kilimanjaro above sea level?",
    "What currency is used in Switzerland?",
    "Where is the Great Barrier Reef located?",

    # History & Literature (25-30)
    "Who was the first emperor of the Roman Empire?",
    "In which year did the French Revolution start?",
    "Who wrote the epic poem The Odyssey?",
    "What was the significance of the Magna Carta in 1215?",
    "Which countries fought in the Battle of Waterloo?",
    "Who was the prime minister of Britain during World War II?",

    # Sports & Entertainment (31-36)
    "Who won the FIFA Men's World Cup in 2022?",
    "How many grand slam titles has Rafael Nadal won?",
    "What is the plot of the movie Interstellar?",
    "Who composed Beethoven's Symphony No. 9?",
    "How many players are on the field in an American football team?",
    "Who played Iron Man in the Marvel Cinematic Universe?",

    # External Legal, Finance & Real Estate (37-42)
    "How do I register a limited liability company in California?",
    "What is the formula for calculating compound interest on a mortgage?",
    "How do freelancers calculate income tax in the UK?",
    "What is the current market price of Bitcoin in USD?",
    "How do I invest in index funds through Vanguard?",
    "What are the zoning regulations for residential construction in Texas?",

    # Miscellaneous / Trivia (43-50)
    "Why do flamingos stand on one leg in shallow water?",
    "How many keys are on a standard acoustic grand piano?",
    "What causes the Northern Lights (Aurora Borealis)?",
    "How do noise-canceling headphones filter ambient sound?",
    "What is the lifespan of a giant Galapagos tortoise?",
    "How many milliliters are in a standard fluid ounce?",
    "Why is the sky blue during a clear sunny day?",
    "What is the world record for the 100-meter sprint?"
]

# ==============================================================================
# EDGE CASES & ACRONYM BENCHMARKS
# ==============================================================================
RELEVANT_EDGE_CASES = [
    # Acronyms & Abbreviations
    "wfh?",
    "WFH",
    "wfh policy",
    "pto balance",
    "how many days pto?",
    "lop leave rules",
    "vpn connection guide",
    "pf contribution percentage",
    "ta/da reimbursement rates",
    "byod security guidelines",
    "hr email contact",
    "nda policy requirements",
    
    # Short & Minimal Queries (Single / Double words)
    "leave?",
    "sick leave",
    "maternity",
    "paternity",
    "resignation",
    "wifi password",
    "per diem allowance",
    "gym subsidy",
    "laptop policy",
    "notice period",
    
    # Typos & Conversational Slang
    "can i wrk from hmoe?",
    "sick leav apply procedure",
    "reimbursment for travel",
    "whats the notice period here",
    "can i take unpaid leve?"
]

IRRELEVANT_EDGE_CASES = [
    # External Acronyms & Tech / Finance
    "ROI of NVDA stock",
    "AWS EC2 pricing model",
    "NASA JWST orbital path",
    "NATO member countries",
    "GDP of Japan in 2024",
    "BGP routing table protocol",
    "SQL injection in PHP",
    "LLM fine-tuning using LoRA",
    "TCP 3-way handshake",
    "IP address of Google DNS",
    "ETF vs Mutual Fund",
    "IPO listing process",

    # Short External Trivia / Queries
    "pizza recipe",
    "tesla stock price",
    "weather in tokyo",
    "olympics 2024 winner",
    "who is elon musk?",
    "how to tie a tie",
    "coffee maker repair",
    "how to change car tire",
    "bitcoin halving date",
    "speed of light",

    # Conversational / Nonsense / Out of Scope
    "tell me a funny joke",
    "write a python script for snake game",
    "what is the meaning of life?",
    "can you solve 2x + 5 = 15?",
    "translate hello to spanish"
]


def evaluate_question_set(collection, questions, is_relevant, threshold=THRESHOLD, use_reformulation=True):
    distances = []
    correct_count = 0
    failures = []

    for idx, q in enumerate(questions, 1):
        search_query = reformulate_query(q) if use_reformulation else q
        emb = create_embedding(search_query)
        res = collection.query(query_embeddings=[emb], n_results=1)
        dist = res["distances"][0][0]
        distances.append(dist)

        if is_relevant:
            # Relevant questions should have distance <= threshold
            if dist <= threshold:
                correct_count += 1
            else:
                failures.append((idx, q, search_query, dist))
        else:
            # Irrelevant questions should have distance > threshold
            if dist > threshold:
                correct_count += 1
            else:
                failures.append((idx, q, search_query, dist))

    return distances, correct_count, failures


def run_50_question_benchmark():
    collection = get_collection()
    
    print("=" * 90)
    print("BENCHMARK EVALUATION: STANDARD (50 + 50) & EDGE CASES (ACRONYMS / SHORT QUERIES)")
    print(f"Distance Metric: Cosine | Active Threshold: {THRESHOLD}")
    print("=" * 90)
    
    # 1. Standard 50 Relevant Questions
    print("\n[1/4] Evaluating 50 Standard Relevant Questions...")
    std_rel_dists, std_rel_passes, std_rel_fails = evaluate_question_set(
        collection, RELEVANT_QUESTIONS_50, is_relevant=True, threshold=THRESHOLD
    )
    
    # 2. Standard 50 Irrelevant Questions
    print("[2/4] Evaluating 50 Standard Irrelevant Questions...")
    std_irr_dists, std_irr_blocks, std_irr_leaks = evaluate_question_set(
        collection, IRRELEVANT_QUESTIONS_50, is_relevant=False, threshold=THRESHOLD
    )

    # 3. Relevant Edge Cases & Acronyms
    print(f"[3/4] Evaluating {len(RELEVANT_EDGE_CASES)} Relevant Edge Cases & Acronyms...")
    edge_rel_dists, edge_rel_passes, edge_rel_fails = evaluate_question_set(
        collection, RELEVANT_EDGE_CASES, is_relevant=True, threshold=THRESHOLD
    )

    # 4. Irrelevant Edge Cases & Out-of-Domain Acronyms
    print(f"[4/4] Evaluating {len(IRRELEVANT_EDGE_CASES)} Irrelevant Edge Cases...")
    edge_irr_dists, edge_irr_blocks, edge_irr_leaks = evaluate_question_set(
        collection, IRRELEVANT_EDGE_CASES, is_relevant=False, threshold=THRESHOLD
    )

    # --- RESULTS: STANDARD BENCHMARK ---
    print("\n" + "=" * 90)
    print("1. STANDARD BENCHMARK RESULTS (100 Questions)")
    print("=" * 90)
    print(f"Relevant Questions (Target: PASS <= {THRESHOLD}):")
    print(f"  Passed (True Positives)   : {std_rel_passes} / 50 ({std_rel_passes/50*100:.1f}%)")
    print(f"  Rejected (False Negatives): {len(std_rel_fails)} / 50 ({len(std_rel_fails)/50*100:.1f}%)")
    print(f"\nIrrelevant Questions (Target: REJECT > {THRESHOLD}):")
    print(f"  Blocked (True Negatives)  : {std_irr_blocks} / 50 ({std_irr_blocks/50*100:.1f}%)")
    print(f"  Accepted (False Positives): {len(std_irr_leaks)} / 50 ({len(std_irr_leaks)/50*100:.1f}%)")
    
    std_total_acc = (std_rel_passes + std_irr_blocks) / 100 * 100
    std_error_rate = 100.0 - std_total_acc
    print(f"\n>> Standard Accuracy  : {std_total_acc:.1f}% ({std_rel_passes + std_irr_blocks}/100)")
    print(f">> Standard Error Rate: {std_error_rate:.1f}% ({len(std_rel_fails) + len(std_irr_leaks)} errors)")

    # --- RESULTS: EDGE CASES BENCHMARK ---
    n_edge_rel = len(RELEVANT_EDGE_CASES)
    n_edge_irr = len(IRRELEVANT_EDGE_CASES)
    total_edge = n_edge_rel + n_edge_irr

    print("\n" + "=" * 90)
    print(f"2. EDGE CASES BENCHMARK RESULTS ({total_edge} Questions: Acronyms, Slang, Typos)")
    print("=" * 90)
    print(f"Relevant Edge Cases (Target: PASS <= {THRESHOLD}):")
    print(f"  Passed (True Positives)   : {edge_rel_passes} / {n_edge_rel} ({edge_rel_passes/n_edge_rel*100:.1f}%)")
    print(f"  Rejected (False Negatives): {len(edge_rel_fails)} / {n_edge_rel} ({len(edge_rel_fails)/n_edge_rel*100:.1f}%)")
    print(f"\nIrrelevant Edge Cases (Target: REJECT > {THRESHOLD}):")
    print(f"  Blocked (True Negatives)  : {edge_irr_blocks} / {n_edge_irr} ({edge_irr_blocks/n_edge_irr*100:.1f}%)")
    print(f"  Accepted (False Positives): {len(edge_irr_leaks)} / {n_edge_irr} ({len(edge_irr_leaks)/n_edge_irr*100:.1f}%)")

    edge_total_acc = (edge_rel_passes + edge_irr_blocks) / total_edge * 100
    edge_error_rate = 100.0 - edge_total_acc
    print(f"\n>> Edge Case Accuracy  : {edge_total_acc:.1f}% ({edge_rel_passes + edge_irr_blocks}/{total_edge})")
    print(f">> Edge Case Error Rate: {edge_error_rate:.1f}% ({len(edge_rel_fails) + len(edge_irr_leaks)} errors)")

    # --- OVERALL COMBINED SUMMARY ---
    all_rel_passes = std_rel_passes + edge_rel_passes
    all_rel_total = 50 + n_edge_rel
    all_irr_blocks = std_irr_blocks + edge_irr_blocks
    all_irr_total = 50 + n_edge_irr
    grand_total = all_rel_total + all_irr_total
    grand_correct = all_rel_passes + all_irr_blocks
    grand_accuracy = grand_correct / grand_total * 100
    grand_error_rate = 100.0 - grand_accuracy

    print("\n" + "=" * 90)
    print(f"3. GRAND TOTAL SUMMARY ({grand_total} Questions Evaluated)")
    print("=" * 90)
    print(f"Grand Accuracy  : {grand_accuracy:.1f}% ({grand_correct}/{grand_total} correct)")
    print(f"Grand Error Rate: {grand_error_rate:.1f}% ({grand_total - grand_correct}/{grand_total} errors)")

    # --- FALSE POSITIVES & FALSE NEGATIVES BREAKDOWN ---
    if edge_rel_fails or std_rel_fails:
        print("\n" + "-" * 90)
        print("[!] RELEVANT QUESTIONS REJECTED (False Negatives - dist > threshold):")
        print("-" * 90)
        for idx, q, sq, d in std_rel_fails:
            print(f"  [Standard #{idx:02d}] dist={d:.4f} > {THRESHOLD} | Original: \"{q}\" -> Reformulated: \"{sq}\"")
        for idx, q, sq, d in edge_rel_fails:
            print(f"  [Edge Case #{idx:02d}] dist={d:.4f} > {THRESHOLD} | Original: \"{q}\" -> Reformulated: \"{sq}\"")

    if edge_irr_leaks or std_irr_leaks:
        print("\n" + "-" * 90)
        print("[!] IRRELEVANT QUESTIONS ACCEPTED (False Positives - dist <= threshold):")
        print("-" * 90)
        for idx, q, sq, d in std_irr_leaks:
            print(f"  [Standard #{idx:02d}] dist={d:.4f} <= {THRESHOLD} | Original: \"{q}\" -> Reformulated: \"{sq}\"")
        for idx, q, sq, d in edge_irr_leaks:
            print(f"  [Edge Case #{idx:02d}] dist={d:.4f} <= {THRESHOLD} | Original: \"{q}\" -> Reformulated: \"{sq}\"")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_50_question_benchmark()
