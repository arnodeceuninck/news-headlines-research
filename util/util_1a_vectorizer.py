# The functions used here are based on code written in "1a Vectorizer.ipynb",
#  look at the notebook for extra explanation.

import operator


def show_most_informative_features(model, vectorizer=None, text=None, n=10):
    # Function copied from stackoverflow: https://stackoverflow.com/questions/48401148/document-classification-with-scikit-learn-most-efficient-way-to-get-the-words

    # Extract the vectorizer and the classifier from the pipeline
    if vectorizer is None:
        vectorizer = model.named_steps['vectorizer']
    elif text is not None:
        vectorizer.fit_transform([text])
    else:
        pass # Assuming suplied vectorizer is correct and no text to transform

    classifier = model.named_steps['classifier']
    feat_names = vectorizer.get_feature_names_out()

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {}.".format(
                classifier.__class__.__name__
            )
        )

    # Otherwise simply use the coefficients
    tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], feat_names),
        key=operator.itemgetter(0), reverse=True
    )

    # Get the top n and bottom n coef, name pairs
    topn = zip(coefs[:n], coefs[:-(n + 1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append(
            "Classified as: {}".format(model.predict([text]))
        )
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(
                cp, fnp, cn, fnn
            )
        )

    print("\n".join(output))


# stemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import DutchStemmer

stemmer = DutchStemmer()
analyzer = CountVectorizer().build_analyzer()


def stem_analyzer(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
