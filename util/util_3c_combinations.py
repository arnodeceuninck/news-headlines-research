from util import add_length_to_dataframe, add_headline_embedding_to_dataframe, add_diff_length


def add_extra_features(df):
    added_features = ["Length", "NumWordsDiff", "AvgWordLengthDiff", "MaxWordLengthDiff"]
    df = add_length_to_dataframe(df)
    df = add_diff_length(df)
    df, added_features_embed = add_headline_embedding_to_dataframe(df)
    return df, added_features + added_features_embed


