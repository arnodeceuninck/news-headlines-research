from sentence_transformers import SentenceTransformer


def add_headline_embedding_to_dataframe(df, emb_prefix="Emb", text_column="Headline"):
    bertje = SentenceTransformer("jegormeister/bert-base-dutch-cased-snli")

    embedding_columns = list(map(lambda i: f"{emb_prefix}{i}", range(768)))
    added_features = embedding_columns

    modified_df = df.copy()
    modified_df[embedding_columns] = bertje.encode(df[text_column].tolist(), show_progress_bar=True)
    return modified_df, added_features
