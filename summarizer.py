import nltk
import networkx as nx
import itertools
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cosine as cosine_similarity #0 bad, 1 good
from scipy.misc import comb # n choose k
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from nltk.corpus import reuters

def normalize_and_tokenize(text, stemmer = nltk.PorterStemmer().stem):
    """
    Alternateively, try the slower: 
        stemmer = nltk.WordNetLemmatizer().lemmatize
    Or getting really (unnecessarily!) fancy: 
        stemmer = lambda t: nltk.PorterStemmer().stem(nltk.WordNetLemmatizer().lemmatize(t))
    """
    tokens = word_tokenize(text)
    return [stemmer(t).translate(None, string.punctuation) for t in tokens]


def fit_vectorizer(text, verbose=False, tfidf=False):
    if tfidf:
        vocab = word_tokenize(text)
        vocab = set(vocab)
        vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', 
                   stop_words='english', vocabulary=vocab, lowercase=True)
        token_dict = {}
        for article in reuters.fileids():
            token_dict[article] = reuters.raw(article)
        for k in token_dict.keys():
            try:
                vect.fit(token_dict[k])
            except Exception, e:
                if verbose:
                    print k, e
    else:
        vect = CountVectorizer(tokenizer=word_tokenize, lowercase=True, preprocessor=None, stop_words='english', decode_error='ignore')
    return vect
    
    
def sparse_cosine_similarity_matrix(sp_mat):
    """Returns the distance matrix of an input sparse matrix using cosine distance"""
    n = sp_mat.shape[0]
    k = int(comb(n,2)) + 1 # not sure why this is off by one...
    dx = np.empty(k)
    for z, ij in enumerate(itertools.combinations(range(n),2)):
        i,j = ij
        u,v = sp_mat.getrow(i), sp_mat.getrow(j)
        dx[z] = cosine_similarity(u.todense(), v.todense())
    return 1-dx
    
def similarity_graph_from_sparse_matrix(sp_mat):
    n = sp_mat.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i,j in itertools.combinations(range(n),2):
        u,v = sp_mat.getrow(i), sp_mat.getrow(j)
        d = cosine_similarity(u.todense(), v.todense())
        g.add_edge(i,j, {'weight':d})
    return g
    
def summarize(text, n=5, tfidf=False):
    """
    Given an input document, extracts and returns representative sentences.
    At present, returns top n sentences, but I hope to find an unsupervised 
    heuristic to determine an appropriate n.
    """
    print "Reading document..."
    sentences = sent_tokenize(text)
    print "Fitting vectorizer..."
    vectorizer = fit_vectorizer(text, tfidf=tfidf)
    if tfidf:
        print "tf-idf transforming sentences..."
        term_doc_matrix = vectorizer.transform(sentences)
    else:
        term_doc_matrix = vectorizer.fit_transform(sentences)
    print "Building similarity graph..."
    g = similarity_graph_from_sparse_matrix(term_doc_matrix)
    print "Calculating sentence pagerank (lexrank)..."
    scores = pd.Series(nx.pagerank(g, weight='weight'))
    scores.sort(ascending=False)
    ix = pd.Series(scores.index[:n])
    ix.sort()
    summary = [sentences[i] for i in ix]
    return {'g':g, 'scores':scores, 'tdm':term_doc_matrix, 'summary':summary}

# Train a tfidf vectorizer on the brown corpus

#vocabulary = brown.words(categories='news')
#vocabulary.extend(word_tokenize(text))

        
    
# From TwitterProject2/outlier_analysis.py
#vectorizer = CountVectorizer(tokenizer=None, preprocessor=None, stop_words=None, decode_error='ignore')

if __name__ == '__main__':
    text = """
    This is the first sentence. I'm not really writing an article about anything. I just need a few sentences to play with.
    So far I have like, what three? I guess that last one was the fourth sentence making this my fifth sentence. Quite
    an interesting article I'm writing here. Woohoo. Look at me go. Maybe just a few more sentences.
    """