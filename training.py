import re
from datasets import load_dataset
from gensim.models import Word2Vec
from tqdm import tqdm

# 1. Text streamer class
class GutenbergStreamer:
    def __init__(self, max_blocks=100000):
        self.max_blocks = max_blocks
        self.dataset = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)

    def __iter__(self):
        # Progress bar
        pbar = tqdm(total=self.max_blocks, desc="Streaming & Training")

        for i, row in enumerate(self.dataset):
            if i >= self.max_blocks:
                break

            # Pull the text from the incoming data
            text = row["TEXT"].strip()
            if text:
                # Regex tokenization
                tokens = re.findall(r'\b\w+\b', text.lower())
                if tokens:
                    yield tokens

            pbar.update(1)
        pbar.close()

def train_gutenberg_generator():
    print("Initializing Project Gutenberg stream...")
    # Initialize streaming object
    streamer = GutenbergStreamer(max_blocks=50000)

    print("Training Word2Vec model...")
    # Passing the streamer object directly to sentences
    model = Word2Vec(
        sentences=streamer,
        vector_size=200,
        window=7,
        min_count=3,
        workers=4
    )

    return model

# Training
gutenberg_model = train_gutenberg_generator()

# Save the model
model_filename = "gutenberg_novel_thesaurus.model"
print(f"Saving trained model to {model_filename}...")
gutenberg_model.save(model_filename)
print("Model saved successfully.")
