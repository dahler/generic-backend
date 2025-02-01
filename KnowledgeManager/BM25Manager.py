from rank_bm25 import BM25Okapi
from stopwords import get_stopwords
# from sklearn.preprocessing import MinMaxScaler
# import nltk
# import spacy
# nlp = spacy.blank("id")
# nltk.download('punkt')
# nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re

# Initialize Sastrawi Stemmer & Stopword Remover
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()


class BM25Manager: 

    def getBM25Score(self, query, doc_texts):
        # Get the list of stopwords
        # stop_words = set(get_stopwords('id'))
        # print(stop_words)
        
        # Tokenize the documents and remove stopwords
        tokenized_corpus = [
            # [word for word in nltk.word_tokenize(doc.lower()) if word not in stop_words]
            self.tokenize_and_remove_stopwords(doc) for doc in doc_texts
        ]

        # tokenized_corpus = self.tokenize_and_remove_stopwords(doc_texts)
        query_keywords = self.tokenize_and_remove_stopwords(query)
        
        # Initialize BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Tokenize the query and remove stopwords
        # query_keywords = [word for word in nltk.word_tokenize(query.lower()) if word not in stop_words]
        
        # Get BM25 scores
        bm25_scores = bm25.get_scores(query_keywords)

        return bm25_scores
    
    def tokenize_and_remove_stopwords(self, text):
        """Tokenizes, removes stopwords, and stems words using Sastrawi"""
        # print(text)
        text = text.lower()
        text = stopword_remover.remove(text)  # Remove stopwords
        text = stemmer.stem(text)  # Apply stemming
        tokens = re.findall(r'\b\w+\b', text)  # Extract words
        return tokens