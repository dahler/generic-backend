import re
from rank_bm25 import BM25Okapi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import time

# Initialize Sastrawi Stemmer & Stopword Remover once (avoids redundant calls)
stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())  # Store in set for O(1) lookup

class BM25Manager:
    def getBM25Score(self, query, doc_texts):
        """Compute BM25 scores for a query against a corpus."""
        # Tokenize documents and stem selectively
        tokenized_corpus = [self.tokenize_and_process(doc, return_list=True) for doc in doc_texts]

        # Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus)

        # Tokenize query
        query_keywords = self.tokenize_and_process(query, return_list=True)

        # Get BM25 scores
        return bm25.get_scores(query_keywords)

    def tokenize_and_process(self, text, return_list=False):
        """Tokenizes, removes stopwords, and stems words efficiently, applying stemming only when necessary."""
        text = text.lower()
        
        start_time = time.time()
        # Tokenize and filter stopwords
        processed_tokens = [word for word in re.findall(r'\b\w+\b', text) if word not in stopwords]

        end_time = time.time()
        time_taken = end_time - start_time
        print("Time Taken to tokenize :", time_taken)
        
        start_time = time.time()
        # Apply stemming only when necessary (e.g., word length > 2, as short words might be roots already)
        # processed_tokens = [stemmer.stem(word) if len(word) > 2 else word for word in tokens]

        end_time = time.time()
        time_taken = end_time - start_time
        print("Time Taken to stem:", time_taken)

        return list(processed_tokens) if return_list else " ".join(processed_tokens)  # Return list or string
