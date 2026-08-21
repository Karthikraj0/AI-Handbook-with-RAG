import numpy as np
from rag.vectorstore import get_collection
from rag.embeddings import create_embedding

RELEVANT_QUESTIONS = [
    # Leave Policy
    "How many days of annual leave am I entitled to?",
    "What is the procedure for applying for sick leave?",
    "Can maternity leave be extended?",
    # WFH Policy
    "How many days per week can employees work from home?",
    "What equipment does the company provide for remote work?",
    # IT Security Policy
    "How often must I change my company account password?",
    "Can I connect my personal USB drive to my work laptop?",
    # Attendance Policy
    "What are the official core working hours of the office?",
    "What happens if an employee is consistently late?",
    # Reimbursement & Travel Policy
    "What is the maximum daily food allowance for business travel?",
    "How do I submit an expense report for reimbursement?",
    # Benefits
    "What health insurance benefits does the company provide?",
]

IRRELEVANT_QUESTIONS = [
    # General Trivia & Science
    "What is the distance from the Earth to the Moon?",
    "How do you bake a chocolate chip cookie?",
    "What is the speed of light in a vacuum?",
    "Who won the FIFA World Cup in 2022?",
    # Unrelated Programming / Tech
    "How do I sort a binary search tree in C++?",
    "What is the difference between TCP and UDP protocols?",
    # Pop Culture / Random
    "What is the plot of the movie Inception?",
    "Where is the best tourist destination in Paris?",
    # Out-of-Scope Enterprise / External
    "What is the current stock price of Apple Inc?",
    "How do I file taxes in Germany as a freelancer?",
]


def test_distance_benchmark():
    collection = get_collection()
    
    print("=" * 85)
    print(f"{'QUERY':<52} | {'BEST DIST':<10} | {'TOP 3 DISTANCES'}")
    print("=" * 85)
    
    relevant_best_dists = []
    print("\n--- RELEVANT QUESTIONS (In Company Handbook) ---")
    for q in RELEVANT_QUESTIONS:
        emb = create_embedding(q)
        res = collection.query(query_embeddings=[emb], n_results=3)
        distances = res["distances"][0]
        best_d = distances[0]
        relevant_best_dists.append(best_d)
        top3_str = ", ".join([f"{d:.4f}" for d in distances])
        print(f"{q[:50]:<52} | {best_d:<10.4f} | [{top3_str}]")
        
    irrelevant_best_dists = []
    print("\n--- IRRELEVANT QUESTIONS (Out-of-Scope / General) ---")
    for q in IRRELEVANT_QUESTIONS:
        emb = create_embedding(q)
        res = collection.query(query_embeddings=[emb], n_results=3)
        distances = res["distances"][0]
        best_d = distances[0]
        irrelevant_best_dists.append(best_d)
        top3_str = ", ".join([f"{d:.4f}" for d in distances])
        print(f"{q[:50]:<52} | {best_d:<10.4f} | [{top3_str}]")
        
    print("\n" + "=" * 85)
    print("STATISTICAL SUMMARY (Cosine Distance)")
    print("=" * 85)
    
    rel_min = min(relevant_best_dists)
    rel_max = max(relevant_best_dists)
    rel_mean = np.mean(relevant_best_dists)
    
    irrel_min = min(irrelevant_best_dists)
    irrel_max = max(irrelevant_best_dists)
    irrel_mean = np.mean(irrelevant_best_dists)
    
    gap = irrel_min - rel_max
    
    print(f"Relevant Questions  : Min = {rel_min:.4f} | Mean = {rel_mean:.4f} | Max = {rel_max:.4f}")
    print(f"Irrelevant Questions: Min = {irrel_min:.4f} | Mean = {irrel_mean:.4f} | Max = {irrel_max:.4f}")
    print(f"Distance Gap (Irrel Min - Rel Max): {gap:.4f}")
    
    if gap > 0:
        recommended_threshold = (rel_max + irrel_min) / 2
        print(f"\n=> Clear separation found! Recommended DISTANCE_THRESHOLD: {recommended_threshold:.2f}")
    else:
        recommended_threshold = (rel_mean + irrel_mean) / 2
        print(f"\n=> Suggested balanced DISTANCE_THRESHOLD: {recommended_threshold:.2f}")
    print("=" * 85)


if __name__ == "__main__":
    test_distance_benchmark()
