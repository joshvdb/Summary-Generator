import networkx as nx
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA, TruncatedSVD, NMF

# define stop words list
stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
              'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
              'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
              'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don',
              'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
              'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
              'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
              'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
              'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 'being',
              'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']


def tokenize_text(document, nlp):
    """
    Function to tokenize text - the string is split into a list of individual words.

    :param document: str
    :param nlp: spacy nlp object
    :return: [str]
    """

    return [token.text for token in nlp(document)]


def remove_stop_words(text_tokens):
    """
    Function to remove all stop-words from our text.

    :param text_tokens: [str]
    :return: [str]
    """

    return [words for words in text_tokens if words not in stop_words]


def remove_non_words(text_tokens):
    """
    Function to remove all non-words (such as special characters) from our text.

    :param text_tokens: [str]
    :return: [str]
    """

    # define the set of Latin characters
    keep_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z']

    return [words for words in text_tokens if all(letter in keep_list for letter in list(words)) is True]


def convert_to_string(tokens):
    """
    Function to join together a list of words into a string. This can be used to reconstitute text after stemming or
    lemmatization, for example.

    :param tokens: [str]
    :return: str
    """

    return ' '.join(tokens)


def lemmatize(text, nlp):
    """
    Function to lemmatize all words in our text.

    :param text: str
    :param nlp: spacy nlp token
    :return: [str]
    """

    return [word.lemma_ for word in nlp(text)]


def get_topics(model, nlp_model, n_top_words):
    """
    Function to obtain the topics from a Topic Model.

    :param model: Scikit-learn model object
    :param nlp_model: Scikit-learn vectorizer model object
    :param n_top_words: int
    :return: [str]
    """

    words = nlp_model.get_feature_names()

    return [convert_to_string([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]) for topic_idx, topic in enumerate(model.components_)]


def text_rank(sentence_vector_list, number_topics, sentences):
    """
    Function to obtain the most relevant sentences in a text using the TextRank method.

    :param sentence_vector_list: [float]
    :param number_topics: int
    :param sentences: [str]
    :return: [str]
    """

    nx_graph = nx.from_numpy_array(cosine_similarity(sentence_vector_list, sentence_vector_list))
    scores = nx.pagerank(nx_graph)

    summary = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)[0:number_topics]

    return list(s for score, i, s in summary)


def get_sentences(text, nlp):
    """
    Function to obtain a list of sentences, and a list of sentence vectors. The outputs of this function are used as
    input to the TextRank model.

    :param text: str
    :param nlp: spacy nlp object
    :return: [str], [float]
    """

    # get sentences from text
    sentences = [sentence for sentence in
                 text.replace('!', '.').replace('?', '.').split('.')]

    processed_sentences = [convert_to_string(remove_junk(tokenize_text(sentence, nlp))) for sentence in
                 text.replace('!', '.').replace('?', '.').split('.')]

    # convert the sentences into a list of document vectors
    sentence_vector_list = [nlp(sentence).vector for sentence in processed_sentences]

    return sentences, sentence_vector_list


def remove_junk(text_tokens):
    """
    Function to remove all junk characters from a string.

    :param text_tokens: [str]
    :return: [str]
    """

    # define the set of Latin characters
    keep_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ',', "'"]

    return [words for words in text_tokens if all(letter in keep_list for letter in list(words)) is True]


def full_summarizer_word_comparison(sentences, topic_sentences, number_topics):
    """
    Function to obtain a summary from a larger text, based on the similarity between the sentences of the text and the
    list of topic sentences, obtained from a summary generation method.

    :param sentences: [str]
    :param topic_sentences: [str]
    :param number_topics: int
    :return: [str]
    """

    word_counts = []

    for sentence in sentences:
        document_1_words = sentence.split()
        document_2_words = ''.join(topic_sentences).split()

        common_words = set(document_1_words).intersection(set(document_2_words))
        word_counts.append(len(common_words))

    return [j for i, j in sorted(list(zip(word_counts, sentences)), reverse=True)][0:number_topics]


def get_summary_model(processed_text, model_type, number_topics):
    """
    Function to train a Topic Model (LDA, LSA or NMF) and an associated NLP model (TF or TF-IDF).

    Topic Models Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

    NLP Models Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

    :param processed_text: [str]
    :param model_type:  str (the type of Topic Model to use)
    :param number_topics: int
    :return: Scikit-learn vectorizer model object, Scikit-learn model object
    """

    if model_type == 'LDA':
        count_model = CountVectorizer(ngram_range=(1, 1)).fit(processed_text)
        return count_model, LDA(n_components=number_topics, learning_method='batch').fit(count_model.fit_transform(processed_text))
    if model_type == 'LSA':
        tf_idf_model = TfidfVectorizer(ngram_range=(1, 1)).fit(processed_text)
        return tf_idf_model, TruncatedSVD(n_components=number_topics, algorithm='randomized', n_iter=100, random_state=122).fit(tf_idf_model.transform(processed_text))
    else:
        tf_idf_model = TfidfVectorizer(ngram_range=(1, 1)).fit(processed_text)
        return tf_idf_model, NMF(n_components=number_topics, init='random', random_state=0).fit(tf_idf_model.transform(processed_text))


def get_summary_from_model(text, model_type, number_words, target_percentage, nlp):
    """
    Function to generate a summary using a Topic Model method (LDA, LSA or NMF).

    :param text: str
    :param model_type: str (the type of Topic Model to use)
    :param number_words: int
    :param target_percentage: float the target length of the summary (by percentage of the total text length) (between 0 and 1)
    :param nlp: spacy nlp object
    :return: str
    """

    # get sentences from text
    sentences = [sentence for sentence in
                 text.lower().replace('!', '.').replace('?', '.').split('.')]

    # this pre-processing removes junk characters, and converts all sentence-ending characters to full-stops, for the
    # purpose of splitting the text into sentences
    processed_text = [convert_to_string(
        lemmatize(convert_to_string(remove_non_words(remove_stop_words(tokenize_text(sentence, nlp)))), nlp)) for
        sentence in sentences]

    # begin generating topics from the text, beginning with 1 topic
    number_topics = 1
    nlp_model, summary_model = get_summary_model(processed_text, model_type, number_topics)
    topic_sentences = get_topics(summary_model, nlp_model, number_words)
    percentage = len('. '.join(full_summarizer_word_comparison(sentences, topic_sentences, number_topics))) / len(text)

    # check if the ratio of the summary to the text is below the user-defined threshold
    while percentage < target_percentage:
        number_topics += 1
        nlp_model, summary_model = get_summary_model(processed_text, model_type, number_topics)
        topic_sentences = get_topics(summary_model, nlp_model, number_words)
        percentage = len('. '.join(full_summarizer_word_comparison(sentences, topic_sentences, number_topics))) / len(text)

    return '. '.join(full_summarizer_word_comparison(sentences, topic_sentences, number_topics))


def get_summary_from_text_rank(text, target_percentage, nlp):
    """
    Function to generate a summary using the TextRank method (an Indicator Representation method).

    :param text: str
    :param target_percentage: float (the target length of the summary - by percentage of the total text length) (between 0 and 1)
    :param nlp: spacy nlp object
    :return: str
    """

    # general pre-processing
    sentences, sentence_vector_list = get_sentences(text, nlp)

    # begin generating topics from the text, beginning with 1 topic
    number_topics = 1
    percentage = len('. '.join(text_rank(sentence_vector_list, number_topics, sentences))) / len(text)

    # check if the ratio of the summary to the text is below the user-defined threshold
    while percentage < target_percentage:
        number_topics += 1
        percentage = len('. '.join(text_rank(sentence_vector_list, number_topics, sentences))) / len(text)

    return '. '.join(text_rank(sentence_vector_list, number_topics, sentences))


def get_summaries(text, number_words, target_percentage, nlp):
    """
    Function to generate summaries of an input text using LDA, LSA, NMF and TextRank.

    :param text: str
    :param number_words: int (the maximum number of words to use for each topic)
    :param target_percentage: float (the target length of the summary - by percentage of the total text length) (between 0 and 1)
    :param nlp: spacy nlp object
    :return: [(str, str)]
    """

    text_rank_summary = get_summary_from_text_rank(text, target_percentage, nlp)
    lda_summary = get_summary_from_model(text, 'LDA', number_words, target_percentage, nlp)
    lsa_summary = get_summary_from_model(text, 'LSA', number_words, target_percentage, nlp)
    nmf_summary = get_summary_from_model(text, 'NMF', number_words, target_percentage, nlp)

    return [('TextRank', text_rank_summary), ('LDA', lda_summary), ('LSA', lsa_summary), ('NMF', nmf_summary)]


def print_summaries(summaries):
    """
    Function to print out a list of summaries.

    :param summaries: [(str, str)]
    :return:
    """

    for method, summary in summaries:
        print(method)
        print('')
        print(summary)
        print('')


def get_diff(text_1, text_2):
    """
    Function to get the percentage difference between two strings, for the purpose of comparing two summaries.

    :param text_1: str
    :param text_2: str
    :return: float
    """

    return str(round(SequenceMatcher(None, text_1, text_2).ratio()*100, 2)) + '%'
