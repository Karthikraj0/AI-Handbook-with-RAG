import numpy as np
from rag.vectorstore import get_collection
from rag.embeddings import create_embedding

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


def run_50_question_benchmark():
    collection = get_collection()
    
    print("=" * 90)
    print(f"BENCHMARK EVALUATION: 50 RELEVANT vs 50 IRRELEVANT QUESTIONS")
    print(f"Distance Metric: Cosine | Active Threshold: {THRESHOLD}")
    print("=" * 90)
    
    # 1. Evaluate Relevant Questions
    relevant_distances = []
    relevant_passes = 0
    relevant_failures = []
    
    print("\nProcessing 50 Relevant Questions...")
    for idx, q in enumerate(RELEVANT_QUESTIONS_50, 1):
        emb = create_embedding(q)
        res = collection.query(query_embeddings=[emb], n_results=1)
        dist = res["distances"][0][0]
        relevant_distances.append(dist)
        
        passed = dist <= THRESHOLD
        if passed:
            relevant_passes += 1
        else:
            relevant_failures.append((idx, q, dist))
            
    # 2. Evaluate Irrelevant Questions
    irrelevant_distances = []
    irrelevant_blocks = 0
    irrelevant_leaks = []
    
    print("Processing 50 Irrelevant Questions...")
    for idx, q in enumerate(IRRELEVANT_QUESTIONS_50, 1):
        emb = create_embedding(q)
        res = collection.query(query_embeddings=[emb], n_results=1)
        dist = res["distances"][0][0]
        irrelevant_distances.append(dist)
        
        blocked = dist > THRESHOLD
        if blocked:
            irrelevant_blocks += 1
        else:
            irrelevant_leaks.append((idx, q, dist))

    # 3. Print Results & Confusion Matrix
    print("\n" + "=" * 90)
    print("CONFUSION MATRIX & ACCURACY")
    print("=" * 90)
    print(f"Relevant Questions (Target: PASS <= {THRESHOLD}):")
    print(f"  Passed (True Positives)  : {relevant_passes} / 50 ({relevant_passes/50*100:.1f}%)")
    print(f"  Rejected (False Negatives): {len(relevant_failures)} / 50 ({len(relevant_failures)/50*100:.1f}%)")
    
    print(f"\nIrrelevant Questions (Target: REJECT > {THRESHOLD}):")
    print(f"  Blocked (True Negatives) : {irrelevant_blocks} / 50 ({irrelevant_blocks/50*100:.1f}%)")
    print(f"  Accepted (False Positives): {len(irrelevant_leaks)} / 50 ({len(irrelevant_leaks)/50*100:.1f}%)")

    total_accuracy = (relevant_passes + irrelevant_blocks) / 100 * 100
    print(f"\nOVERALL ACCURACY: {total_accuracy:.1f}% ({relevant_passes + irrelevant_blocks}/100 correct)")

    # 4. Statistical Distribution
    rel_d = np.array(relevant_distances)
    irr_d = np.array(irrelevant_distances)

    print("\n" + "=" * 90)
    print("DISTRIBUTION STATISTICS (Cosine Distance)")
    print("=" * 90)
    print(f"{'Category':<22} | {'Min':<8} | {'Median':<8} | {'Mean':<8} | {'Max':<8} | {'Std Dev':<8}")
    print("-" * 90)
    print(f"{'Relevant (50 Qs)':<22} | {rel_d.min():<8.4f} | {np.median(rel_d):<8.4f} | {rel_d.mean():<8.4f} | {rel_d.max():<8.4f} | {rel_d.std():<8.4f}")
    print(f"{'Irrelevant (50 Qs)':<22} | {irr_d.min():<8.4f} | {np.median(irr_d):<8.4f} | {irr_d.mean():<8.4f} | {irr_d.max():<8.4f} | {irr_d.std():<8.4f}")
    print("-" * 90)
    print(f"Distance Gap (Irrel Min - Rel Max): {irr_d.min() - rel_d.max():.4f}")

    if relevant_failures:
        print("\n[!] False Negatives (Relevant Qs rejected):")
        for idx, q, d in relevant_failures:
            print(f"  #{idx:02d}: dist={d:.4f} > {THRESHOLD} -> \"{q}\"")

    if irrelevant_leaks:
        print("\n[!] False Positives (Irrelevant Qs accepted):")
        for idx, q, d in irrelevant_leaks:
            print(f"  #{idx:02d}: dist={d:.4f} <= {THRESHOLD} -> \"{q}\"")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_50_question_benchmark()
