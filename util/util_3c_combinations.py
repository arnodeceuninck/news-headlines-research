from util import add_length_to_dataframe, add_headline_embedding_to_dataframe, add_diff_length
features.append("Length")
train_x_full = add_length_to_dataframe(train_x_full)
test_x = add_length_to_dataframe(test_x)

train_x_full, extra_features = add_headline_embedding_to_dataframe(train_x_full)
test_x, _ = add_headline_embedding_to_dataframe(test_x)
features += extra_features

features.append('NumWordsDiff')
features.append('AvgWordLengthDiff')
features.append('MaxWordLengthDiff')
train_x_full = add_diff_length(train_x_full)
test_x = add_diff_length(test_x)
