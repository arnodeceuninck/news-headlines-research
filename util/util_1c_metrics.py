from sklearn import metrics


def print_evalution(target, predicted):
    cm = metrics.confusion_matrix(target, predicted)
    f1 = metrics.f1_score(target, predicted)

    print(f"f-score: {f1}")
    print(f"Confusion matrix: (TP: {cm[0, 0]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}, TN: {cm[1, 1]})")
    print(f"Accuracy={cm[0, 0] + cm[1, 1]/(cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]):.2f}% ({cm[0, 0] + cm[1, 1]}/{cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]})")

    metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
