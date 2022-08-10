from sklearn import metrics


def print_evaluation(target, predicted):
    cm = metrics.confusion_matrix(target, predicted)
    f1 = metrics.f1_score(target, predicted)

    tp, fp, fn, tn = cm.ravel()
    correct = tp + tn
    total = tp + fp + fn + tn

    print(f"f-score: {f1}")
    print(f"Confusion matrix: (TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    print(f"Accuracy={(100 * correct / total):.2f}% ({correct}/{total})")

    metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()


def fit_predict_evaluate(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    print_evaluation(test_y, predictions)
