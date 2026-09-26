# nlp-thesaurus
# Literary Word Generator & NLP Pipeline

The model uses the Gensim Word2Vec library and trains with 5 epochs using sedthh's Project Gutenberg database's full 48,284 paragraphs. To avoid storing the large Gutenberg database all at once, the dataset is streamed from Hugging Face to the model line by line using a text streamer class built from the Datasets library. Within the streamer class, Regex is used to return tokens that have been stripped of whitespace and punctuation and converted to lowercase.

## Key Features
* **Data Streaming:** Implemented a custom text streaming class using Python generator expressions (`yield`) to pipe paragraphs line-by-line from Hugging Face directly into the training loop.
* **Regex Tokenization:** Replaced standard heavy NLTK linguistic tokenization with a low-level, compiled Regular Expression (`re.findall`) tokenization layer.
* **Domain-Specific Semantics:** Trained using the **Skip-gram architecture** across the Project Gutenberg English dataset for generating expressive and classic words.
* **Instant Offline Inferences:** Serializes weights into binary `.model` files allowing for completely offline user interactive CLI lookups.

---

## System Architecture

```text
[Hugging Face Stream] 
       ↓ (Lazy Evaluation / Yield Iterator)
[Regex Text Normalizer] 
       ↓ (Punctuation Stripping & Lowercasing)
[Gensim Word2Vec Model] → (5 Epoch Multi-Pass) → [Local Binary (.model)]
                                                          ↓
                                                    [Offline CLI Tool]
```

---

## Tech Stack
* **Language:** Python 3.11+
* **Machine Learning Framework:** Gensim (Word2Vec)
* **Data Pipelines:** Hugging Face Datasets
* **Performance Tooling:** Python Regex (`re`), NLTK (Benchmark Baseline), TQDM

---

## Execution

### 1. Training the Pipeline
Run the main pipeline to stream the corpus, normalize the text, and serialize the mathematical vector map:
```bash
python training.py
```

### 2. Querying the Model
Once training completes and saves your local weights, run the lightning-fast interactive CLI workspace to search for novel-writing inspiration:
```bash
python run.py
```

*Example Inference Output:*
```text
Enter a word: melancholy

Similarities for  "melancholy":
  -> sadness (confidence: 0.75)
  -> somber (confidence: 0.74)
  -> ghastly (confidence: 0.73)
  -> gloomy (confidence: 0.71)
...
```

---

## 📚 References & Data Citations
* **Corpus Data:** [Project Gutenberg English Language eBooks Dataset](https://huggingface.co) via Richard Nagyfi.
* **Vector Engine:** Rehurek, R., & Sojka, P. (2010). *Gensim: Topic Modelling for Humans.*
* **Pipeline Infrastructure:** Lhoest, Q., et al. (2021). *Datasets: A Community Library for Natural Language Processing.*
