import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from nltk.corpus import stopwords
# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import pickle
# %matplotlib inline

class CustomLDA:

    stop_words = stopwords.words('english') +\
                ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want',\
                 'come', 'take', 'use', 'would', 'can','one', 'mr', 'bbc', 'image', 'getty',\
                 'de', 'en', 'caption', 'also', 'copyright', 'something', 'http' , 'https']
    
    def __init__(self):
        
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.model = None
        self.data_lemmatized = None
        self.data_vectorized = None
        self.vectorizer = None
        
        # initializa factorizer
        self.vectorizer = CountVectorizer(analyzer='word',       
                                     min_df=10,# minimum read occurences of a word 
                                     stop_words=CustomLDA.stop_words, # remove stop words
                                     lowercase=True,# convert all words to lowercase
                                     token_pattern='[a-zA-Z0-9]{3,}', # num chars > 3
                                     # max_features=50000, # max number of uniq words    
                                    )
        
    def build_model(self, data=None,_model=None, load_model=False):
           
        if load_model:
            self.load_model()
        else:
            # data is a list of sentences
            data = self.clean_data(data)
            self.data = data
            self.data_words = self.sent_to_words(data)
            self.data_lemmatized = self.lemmatize(self.data_words, allowed_postags=['NOUN', 'VERB']) #select noun and verb
            self.data_vectorized = self.vectorize(self.data_lemmatized)
            assert _model is not None, "load_model is False and no model was provided"
            self.find_best_model(_model)
        

    def vectorize(self, data_lemmatized):
        data_vectorized = self.vectorizer.fit_transform(data_lemmatized)
        return data_vectorized
    
    def find_best_model(self, _model, print_stats=True):
        # Define Search Param
        search_params = {'n_components': [10], 'learning_decay': [.2,.4, .5],'learning_offset':[7],
                        'max_iter':[4,6]} 
        
        # Init the Model
#         lda = LatentDirichletAllocation(max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
        # Init Grid Search Class
        model = GridSearchCV(_model, param_grid=search_params)
        # Do the Grid Search
        model.fit(self.data_vectorized)
        
        self.model = model
        # Best Model
        best_lda_model = model.best_estimator_
        # Model Parameters
        print("Best Model's Params: ", model.best_params_)
        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)
        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(self.data_vectorized))
        
    
    def clean_data(self, data):
        data = [self.filter_text(sentence) for sentence in data]
        return data
        
    def filter_text(self, text):
        # Remove urls
        text = re.sub("(?P<url>https?://[^\s]+)", '', text)
        text = re.sub("http", '', text)
        text = re.sub(r'//t\.co.+', '', text)
        # Remove retweets
        text = re.sub(r'^RT @.+\:', '', text)
        # Remove new line characters
        text = re.sub(r'\s+', ' ', text)
        # Remove distracting single quotes
        text = re.sub(r"\'", "", text)
        return text
        
    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
            
    def lemmatize(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #'NOUN', 'ADJ', 'VERB', 'ADV'
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out
    
    def color_green(self, val):
        color = 'green' if val > .1 else 'black'
        return 'color: {col}'.format(col=color)
    
    def make_bold(self, val):
        weight = 700 if val > .1 else 400
        return 'font-weight: {weight}'.format(weight=weight)
    
    def document_topic_matrix(self,data=None,data_vectorized=None, model=None):
        if data is None or data_vectorized is None:
            data = self.data
            data_vectorized = self.data_vectorized
        
        if model is None:
            model = self.model.best_estimator_
            
        self.lda_output = model.transform(data_vectorized)
        # column names
        self.topicnames = ['Topic' + str(i) for i in range(model.n_components)]
        # index names
        docnames = ['Doc' + str(i) for i in range(len(data))]
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(self.lda_output, 2), columns=self.topicnames, index=docnames)
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic['dominant_topic'] = dominant_topic
        # Apply Style
#         df_document_topics = df_document_topic.head(15)
        df_document_topic = df_document_topic.head(50).style.applymap(self.color_green).applymap(self.make_bold)
#         df_document_topics
        return df_document_topic
    
    
    def topic_keyword_matrix(self):
        df_topic_keywords = pd.DataFrame(self.model.best_estimator_.components_)
        # Assign Column and Index
        df_topic_keywords.columns = self.vectorizer.get_feature_names()
        df_topic_keywords.index = self.topicnames
        return df_topic_keywords#.head(20)
    
    # Show top n keywords for each topic
    def show_topics(self, n_words=20):
        '''vectorizer=vectorizer, lda_model=lda_model,'''
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in self.model.best_estimator_.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
            
        # Topic - Keywords Dataframe
        topic_keywords = pd.DataFrame(topic_keywords)
        topic_keywords.columns = ['Word '+str(i) for i in range(topic_keywords.shape[1])]
        topic_keywords.index = ['Topic '+str(i) for i in range(topic_keywords.shape[0])]
        return topic_keywords
    
    def predict_topic(self, text):
        text_2 = list(self.sent_to_words(text))
        text_3 = self.lemmatize(text_2)
        text_4 = self.vectorizer.transform(text_3)
        topic_probability_scores = self.model.best_estimator_.transform(text_4)
        df_topic_keywords = self.show_topics()
        topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()
        infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]

        #topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
        return infer_topic, topic, topic_probability_scores
    
    def save_model(self):
        pickle.dump(self.model, open("custom_lda_trained.model", 'wb'))
        pickle.dump(self.data_lemmatized, open("custom_lda_lemmatizations.model", 'wb'))
        pickle.dump(self.data_vectorized, open("custom_lda_vectorizations.model", 'wb'))
        pickle.dump(self.vectorizer, open("custom_lda_vectorizer.model", 'wb'))
    
    def load_model(self):
        self.model = pickle.load(open("custom_lda_trained.model", 'rb'))
        self.data_lemmatized = pickle.load(open("custom_lda_lemmatizations.model", 'rb'))
        self.data_vectorized = pickle.load(open("custom_lda_vectorizations.model", 'rb'))
        self.vectorizer = pickle.load(open("custom_lda_vectorizer.model", 'rb'))
        