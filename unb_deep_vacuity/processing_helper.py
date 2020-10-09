import json
from os.path import join, dirname, abspath

import numpy as np
import pandas  as pd
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import vstack



def json_to_pd(r_path):
    dic = json.load(open(r_path, 'r'))
    cols = list(dic['train'][0])+['split']
    data = np.concatenate((
        np.array([np.array(list(i.values())+['train']) for i in dic['train']]),
        np.array([np.array(list(i.values())+['test']) for i in dic['test']])))
    ind = [i['id'] for i in dic['train']] + [i['id'] for i in dic['test']]
    json_df = pd.DataFrame(data=data, index=ind, columns=cols)
    json_df['risco'] = pd.to_numeric(json_df['risco'])
    json_df['date'] = pd.to_datetime(json_df['date'])
    json_df = json_df.filter(['txt', 'risco', 'split'])
    return json_df
# Entrada: Caminho até o arquivo e nome do arquivo
# Saída: hdf/dataframe com vetores de frequência, risco e split
# 


def text_to_tfidf_vectors(filename: str, path_to_folder=None):
    """ Receives a filename for json file and a path to folder.
        Returns a dataframe with text vectors instead of
        If no path_to_folder is given, the default folder is inside
        resources/data.
    """
    if not path_to_folder:
        path_to_folder = join(abspath(dirname(__file__)), 'resources/data/')

    VECTOR_MODEL_NAME = "pt_core_news_sm"
    NLP_SPACY = spacy.load(VECTOR_MODEL_NAME)
    TARGET_VARIABLE = "RISCO"
    TEXT_VARIABLE = "TXT"

    path_to_file = path_to_folder + filename + ".json"

    data_df = json_to_pd(path_to_file)


    # Renaming the columns
    # Let's start uppercasing all column names and target variable values
    data_df.columns = map(lambda x: str(x).upper(), data_df.columns)
    data_df[TARGET_VARIABLE] = data_df[TARGET_VARIABLE].apply(
        lambda x: str(x))

    print(data_df.head())

    # Removing ponctuation and stopwords
    # As we can see, we have a lot of tokens from text variable being
    # ponctuations or words that don't have by themselves much meaning.
    # We're going to load a built-in stopwords list to remove these
    # unnecessary tokens.
    stopwords_set = set(STOP_WORDS).union(
        set(stopwords.words('portuguese'))).union(
            set(['anos', 'ano', 'dia', 'dias', 'nº', 'n°']))

    # Removing HTML
    data_df['TXT'] = data_df['TXT'].str.replace(r'<.*?>', '')

    # Lemmatizing and stemming
    print("This is the stopword list: ", sorted(list(stopwords_set)))

    ''' Not all variables are being undestood as strings so we have to force it'''
    preprocessed_text_data = data_df[TEXT_VARIABLE].to_list()
    ''' Create the pipeline 'sentencizer' component '''
    sentencizer = NLP_SPACY.create_pipe('sentencizer')
    try:
        ''' We then add the component to the pipeline if we hadn't done before '''
        NLP_SPACY.add_pipe(sentencizer, before='parser')
    except ValueError:
        print("Pipe already present.")

    print(NLP_SPACY.pipe_names)

    tokenized_data = []
    semantics_data = []
    lemmatized_doc = []
    normalized_doc = []
    raw_doc = []
    for row in preprocessed_text_data:
        doc = NLP_SPACY(row)
        preprocessed_doc = [
            token for token in doc
            if token.is_alpha and token.norm_ not in stopwords_set]
        tokenized_data.append(preprocessed_doc)
        raw_doc.append(" ".join([word.text for word in preprocessed_doc]))
        lemmatized_doc.append(
            " ".join([word.lemma_ for word in preprocessed_doc]))
        normalized_doc.append(
            " ".join([word.norm_ for word in preprocessed_doc]))

    data_df['RAW_DOC'] = raw_doc
    data_df['NORMALIZED_DOC'] = normalized_doc
    data_df['LEMMATIZED_DOC'] = lemmatized_doc

    print(data_df.head())

    # Entity recognition and filtering
    # Some parts of speech may mislead the model associating classes
    # to certain entities that are not really related to the categories.
    processed_tokenized_data = []
    processed_doc_text = []
    entities_obs = []
    entity_unwanted_types = set(['PER', 'ORG'])

    for doc in tokenized_data:
        entities_text = ""
        processed_doc = []
        for token in doc:
            if not token.ent_type_:
                processed_doc.append(token)
            elif token.ent_type_ not in entity_unwanted_types:
                processed_doc.append(token)
                entities_obs.append((token.text, token.ent_type_))
            
        processed_tokenized_data.append(processed_doc)
        processed_doc_text.append(
            " ".join([word.norm_ for word in processed_doc]))

    ''' Processing text on entity level'''
    data_df['PROCESSED_DOC'] = processed_doc_text
    print(data_df.head())

    # Now we're going to remove POS,
    # only allowing proper nouns, nouns, adjectives, adverbs
    # and verb to present in our text variable.

    allowed_pos_set = set(["PROPN", "NOUN", "ADV", "ADJ", "VERB"])

    processed_doc = []
    filtered_token_obs = []
    for doc in processed_tokenized_data:
        doc_tokens = [word for word in doc if str(word.pos_) in allowed_pos_set]
        filtered_token_obs.append(doc_tokens)
        processed_doc.append(" ".join(token.norm_ for token in doc_tokens))

    data_df['PROCESSED_DOC'] = processed_doc
    data_df['TOKENS'] = filtered_token_obs
    print(data_df.head()) 

    # Removing extra spaces originated from the removal of tokens
    space_pattern = r'\s\s+'
    data_df['PROCESSED_DOC'] = data_df['PROCESSED_DOC'].str.replace(space_pattern, " ").str.strip()
    data_df = data_df
    data_df = data_df.drop(columns=['TOKENS']).dropna()
    data_df[TARGET_VARIABLE] = data_df[TARGET_VARIABLE].apply(lambda x: str(x))
    print(data_df.info())

    # Removing accents and symbols
    data_df['TXT'] = data_df['TXT'].str.normalize(
        'NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    print(data_df.info())

    ''' Best parameter using GridSearch (CV score=0.535): 
    {'tfidf__norm': 'l2', 'tfidf__smooth_idf': False, 'tfidf__sublinear_tf': False, 'tfidf__use_idf': True,
    'vect__max_df': 0.2, 'vect__max_features': None, 'vect__min_df': 0.0006, 'vect__ngram_range': (1, 3)}
    Those were obtained on the next code block.
    '''
    count_vectorizer = CountVectorizer(
        max_features=None, min_df=0.0006, max_df=0.2, ngram_range=(1, 3))
    tfidf_transformer = TfidfTransformer(
        norm='l2', use_idf=True, sublinear_tf=False)

    # First let's split train and test data
    train_mask = data_df['SPLIT'] == 'train'
    test_mask = data_df['SPLIT'] == 'test'

    train_df = data_df[train_mask]
    test_df = data_df[test_mask]
    ''' Let's transform the lemmatized documents into count vectors '''
    train_count_vectors = count_vectorizer.fit_transform(
        train_df['PROCESSED_DOC'])
    test_count_vectors = count_vectorizer.transform(
        test_df['PROCESSED_DOC'])

    ''' Then use those count vectors to generate frequency vectors '''
    train_frequency_vectors = tfidf_transformer.fit_transform(
        train_count_vectors)
    test_frequency_vectors = tfidf_transformer.transform(
        test_count_vectors)

    frequency_vectors = vstack((
        train_frequency_vectors, test_frequency_vectors))

    return frequency_vectors
