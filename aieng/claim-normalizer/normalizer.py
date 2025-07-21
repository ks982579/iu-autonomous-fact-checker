import re
from collections import Counter

def extract_search_keywords(claim, max_keywords=5):
    """Extract key terms for NewsAPI search"""
    
    # Remove common stop words and short words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'that', 'this', 'it', 'they', 'them', 'their'
    }
    
    # Clean and split
    words = re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower())
    keywords = [w for w in words if w not in stop_words]
    
    # Simple logic for getting important terms
    # TODO: Improve later!
    word_counts = Counter(keywords)
    top_keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return top_keywords



# ------------- TESTING ---------------

test_easy = [
    "The president announced a new climate policy yesterday.",
    "Apple's stock price increased by 15% last week.", 
    "COVID-19 vaccines are 95% effective against severe illness.",
    "Tesla stock price dropped 20% this week.",
    "Microsoft announced a new AI partnership yesterday.",
    "Biden signed a climate change bill last month.",
    "Apple released a new iPhone model in September.",
    "Meta laid off 10,000 employees in 2024.",
    "Trump is going to deport Elon Musk!",
]

# -------------------------------------

if __name__ == "__main__":
    print("---" * 5 + "Testing" + "---" * 5)
    print("---" * 5 + "EASY" + "---" * 5)
    for claim in test_easy:
        keywords = extract_search_keywords(claim)
        search_query = " ".join(keywords)
        print(f"Claim: {claim}")
        print(f"Search: {search_query}")
        print()