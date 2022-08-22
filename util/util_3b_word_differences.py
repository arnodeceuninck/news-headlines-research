from sentence_transformers import SentenceTransformer

import re


def get_word_sets(test_id, df):
    test_headlines = df[df['Test'] == test_id]['Headline'].values
    test_headlines_word_sets = [set(re.sub('[^a-z0-9 ]+', '', test_headline.lower()).split()) for test_headline in
                                test_headlines]
    return test_headlines_word_sets


def word_in_other_headline(word, word_sets, current_headline_idx):
    for i in range(len(word_sets)):
        if i == current_headline_idx:
            continue
        if word in word_sets[i]:
            return True
    return False


def get_word_differences(word_sets, current_headline_idx):
    word_differences = []
    current_headline_set = word_sets[current_headline_idx]
    for word in current_headline_set:
        if not word_in_other_headline(word, word_sets, current_headline_idx):
            word_differences.append(word)
    return word_differences


def add_difference_strings(df):
    new_df = df.copy()

    current_test = -1
    current_word_set = {}

    new_df.reset_index(drop=True, inplace=True)

    new_df['Word Differences'] = ""

    for i in range(len(new_df)):
        headline_letter = new_df.loc[i, 'Headline ID'][0]
        headline_idx = ord(headline_letter) - ord('A')

        test_id = new_df.loc[i, 'Test']

        if test_id != current_test:
            current_test = test_id
            current_word_set = get_word_sets(test_id, df)

        word_differences = get_word_differences(current_word_set, headline_idx)
        new_df.loc[i, 'Word Differences'] = " ".join(word_differences)

    return new_df


def count_words(word_differences):
    return len(word_differences.split())


def get_avg_word_length(word_differences):
    words_count = count_words(word_differences)
    return len(word_differences) / words_count if words_count > 0 else 0


def get_max_word_length(word_differences):
    words = word_differences.split()
    return max([len(word) for word in words]) if words else 0


def add_diff_length(df):
    new_df = df.copy()

    if not 'Word Differences' in new_df.columns:
        new_df = add_difference_strings(new_df)

    new_df["NumWordsDiff"] = new_df["Word Differences"].apply(count_words)
    new_df["AvgWordLengthDiff"] = new_df["Word Differences"].apply(get_avg_word_length)
    new_df["MaxWordLengthDiff"] = new_df["Word Differences"].apply(get_max_word_length)

    return new_df
