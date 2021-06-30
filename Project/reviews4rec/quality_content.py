import json
import os
from string import punctuation

import numpy as np
import pandas as pd
from collections import Counter
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from readcalc import readcalc
from scipy.stats import entropy
from spellchecker import SpellChecker

# textstat.set_lang("en")
spell = SpellChecker()
DATA_DIR = 'data/'
categories = ['toys', 'kindle', 'movies']
data_file_names = ['Toys_and_Games', 'Kindle_Store', 'Movies_and_TV']


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


def calc_quality_measures(review):
    num_chars, punct_dense, capital_dense, start_with_capital, space_dense, num_misspellings, avg_num_syllable, \
    word_len_entropy, num_words, num_sentences, gunning_fog, flesch_kincaid, smog, pos_entropy, noun_freq, verb_freq = \
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    if len(review) > 0:
        try:
            tokenized_review = word_tokenize(review)
            # Number of characters
            num_chars = len(review)
            # Punctuation
            num_punct = len([ch for ch in review if ch in punctuation])
            punct_dense = num_punct / num_chars
            # Capitalization
            num_capital = len([ch for ch in review if ch.isupper()])
            capital_dense = num_capital / num_chars
            start_with_capital = review[0].isupper()
            # Space Density (percent of all characters)
            num_space = len([ch for ch in review if ch == ' '])
            space_dense = num_space / num_chars
            # Misspellings and typos - number of spelling mistakes
            misspelled = spell.unknown(tokenized_review)
            num_misspellings = len(misspelled)
            # Number of out-of-vocabulary words
            '''
            To identify out-of-vocabulary words, we construct multiple lists of the k most frequent words in Yahoo! Answers, with 
            several k values ranging between 50 and 5000. These lists are then used to calculate a set of “out-of-vocabulary” 
            features, where each feature assumes the list of top-k words for some k is the vocabulary. An example feature created 
            this way is “the fraction of words in an answer that do not appear in the top-1000 words of the collection
            '''
            # Average number of syllables per word
            avg_num_syllable = np.mean([syllable_count(word) for word in tokenized_review])
            # entropy of word lengths
            word_len_entropy = entropy([len(word) for word in tokenized_review])
            # Word length
            num_words = len(tokenized_review)
            # Num sentences
            num_sentences = len(sent_tokenize(review))

            # Readability:
            calc_readability = readcalc.ReadCalc(review)
            # Gunning Fog Index (6-17) 17-difficult, 6-easy
            gunning_fog = calc_readability.get_gunning_fog_index()
            # gunning_fog = textstat.gunning_fog(review)
            # Flesch Kincaid Formula (0-100) 0-difficult, 100-easy
            flesch_kincaid = calc_readability.get_flesch_kincaid_grade_level()
            # flesch_kincaid = textstat.flesch_kincaid_grade(review)
            # SMOG Grading - need at least 30 sentences
            smog = calc_readability.get_smog_index()
            # smog = textstat.smog_index(review)

            # POS - %Nouns, %Verbs
            pos_tags = [item[1] for item in pos_tag(tokenized_review)]
            # Entropy of the part-of-speech tags
            pos_count = list(Counter(pos_tags).values())
            pos_dist = np.array(pos_count) / sum(pos_count)
            pos_entropy = entropy(pos_dist)
            # Formality score - between 0 and 100%, 0 - completely contextualizes language,
            # completely formal language - 100
            noun_freq = len([pos for pos in pos_tags if pos[:2] == 'NN']) / len(tokenized_review)
            # adjective_freq = len([pos for pos in pos_tags if pos[:2] == 'JJ']) / len(tokenized_review)
            # preposition_freq = len([pos for pos in pos_tags if pos[:2] == 'IN']) / len(tokenized_review)
            # article_freq = len([pos for pos in pos_tags if pos[:2] == 'DT']) / len(tokenized_review)
            # pronoun_freq = len([pos for pos in pos_tags if pos[:2] == 'PR']) / len(tokenized_review)
            verb_freq = len([pos for pos in pos_tags if pos[:2] == 'VB']) / len(tokenized_review)
            # adverb_freq = len([pos for pos in pos_tags if pos[:2] == 'RB']) / len(tokenized_review)
            # interjection_freq = len([pos for pos in pos_tags if pos[:2] == 'UH']) / len(tokenized_review)
            # formality_score = (noun_freq + adjective_freq + preposition_freq + article_freq -
            #                    pronoun_freq - verb_freq - adverb_freq - interjection_freq + 100) / 2

        except Exception as e:
            print('Exception: ' + str(e))
            print('Review: ' + str(review))

    return num_chars, punct_dense, capital_dense, start_with_capital, space_dense, num_misspellings, avg_num_syllable, \
           word_len_entropy, num_words, num_sentences, gunning_fog, flesch_kincaid, smog, pos_entropy, noun_freq, \
           verb_freq


for category, file_name in zip(categories, data_file_names):
    print('category: ' + category)
    data_file = os.path.join(DATA_DIR, 'reviews_' + file_name + '_5.json')
    # f = gzip.open(data_file, 'r')
    f = open(data_file, 'r')
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    helpful_votes = []
    total_votes = []

    # i = 0
    try:
        for line in f:
            js = json.loads(line)
            if 'reviewText' not in js:
                js['reviewText'] = ''
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID']))
            items_id.append(str(js['asin']))
            ratings.append(str(js['overall']))
            helpful_votes.append(str(js['helpful'][0]))
            total_votes.append(str(js['helpful'][1]))
            # i += 1
            # if i > 10000:
            #     break
    except Exception as e:
        print(e)

    data = pd.DataFrame({'user_id': pd.Series(users_id),
                         'item_id': pd.Series(items_id),
                         'rating': pd.Series(ratings),
                         'helpful_votes': pd.Series(helpful_votes),
                         'total_votes': pd.Series(total_votes),
                         'review': pd.Series(reviews)})[
        ['user_id', 'item_id', 'rating', 'helpful_votes', 'total_votes', 'review']]

    quality_result = data.apply(lambda x: calc_quality_measures(x['review']), axis=1, result_type='expand')
    quality_result.columns = ['num_chars', 'punct_dense', 'capital_dense', 'start_capital', 'space_dense',
                              'num_misspell', 'avg_num_syllable', 'word_len_entropy', 'num_words', 'num_sentences',
                              'gunning_fog', 'flesch_kincaid', 'smog', 'pos_entropy', '%NN', '%VB']
    data = pd.concat([data, quality_result], axis=1)
    data.to_csv('data/' + category + '_quality.tsv', sep='\t', index=False)
