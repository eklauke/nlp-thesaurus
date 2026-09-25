import sys
from gensim.models import Word2Vec

# 1. Load the model
print("Loading thesaurus...")
try:
    model = Word2Vec.load("gutenberg_novel_thesaurus.model")
    print("Model loaded successfully...")
except FileNotFoundError:
    print("\nError: Could not find \"gutenberg_novel_thesaurus.model\".")
    sys.exit(1)

def find_literary_inspirations(word, limit=40):
    word = word.lower().strip()
    try:
        # 2. Query the vector space
        results = model.wv.most_similar(word, topn=limit)
        print(f"\nSimilarities for \"{word}\":")
        for match, score in results:
            print(f"  -> {match} (confidence: {score:.2f})")
    except KeyError:
        print(f"\nError: \"{word}\" isn't in your classical vocabulary yet.")

# 3. Interface
print("\n" + "="*40)
print("  CLASSIC LITERATURE WORD EXPANDER  ")
print("="*40)
print("Type a word to find creative inspiration. Type \"q\" to exit.")

while True:
    user_input = input("\nEnter a word: ")
    if user_input.strip().lower() == 'q':
        print("Exiting thesaurus...")
        break
    if user_input.strip():
        find_literary_inspirations(user_input)
