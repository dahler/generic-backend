from rank_bm25 import BM25Okapi
from stopwords import get_stopwords
# from sklearn.preprocessing import MinMaxScaler
import nltk
nltk.download('punkt')
# nltk.download('stopwords')

class BM25Manager: 

    def getBM25Score(self, query, doc_texts):
        # Get the list of stopwords
        stop_words = set(get_stopwords('id'))
        # print(stop_words)
        
        # Tokenize the documents and remove stopwords
        tokenized_corpus = [
            [word for word in nltk.word_tokenize(doc.lower()) if word not in stop_words]
            for doc in doc_texts
        ]
        
        # Initialize BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Tokenize the query and remove stopwords
        query_keywords = [word for word in nltk.word_tokenize(query.lower()) if word not in stop_words]
        
        # Get BM25 scores
        bm25_scores = bm25.get_scores(query_keywords)

        return bm25_scores