
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn import preprocessing, linear_model, model_selection, metrics, datasets, base
from matplotlib import pyplot


def make_confusion_matrix(model, x_test, y_test, class_labels,threshold=0.5):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(x_test)[:, 1] >= threshold)
    c_matrix = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(c_matrix, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=class_labels,
           yticklabels=class_labels);
    plt.xlabel('prediction')
    plt.ylabel('actual')
    return plt

def plot_roc(model, x_test, y_test):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(x_test))
    plt.plot(fpr, tpr,lw=2)
    #plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve');
    print("ROC AUC score = ", metrics.roc_auc_score(y_test, model.predict_proba(x_test)[:,1]))
    return plt


def generate_coef_table(feature_names, model, model_type, top_x_features):
    coef_table = pd.DataFrame(list(feature_names), columns=['Variable']).copy()
    if model_type == 'regression':
        feature_scores = model.coef_.transpose()

    elif model_type == 'tree':
        feature_scores = model.feature_importances_

    coef_table.insert(len(coef_table.columns), "Coefs", feature_scores)
    coefficient_table = pd.concat([coef_table.sort_values('Coefs', ascending=False)[:int(top_x_features / 2)],
                                   coef_table.sort_values('Coefs', ascending=False)[-int(top_x_features / 2):]])

    forest_importances = pd.Series(list(coefficient_table['Coefs'][:5]), index=list(coefficient_table['Variable'][:5]))
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Variables")
    ax.set_ylabel("Feature importances")
    fig.tight_layout()

    return coefficient_table, fig

def final_comparison(models, x_test, y_test):
    scores=pd.DataFrame()
    for model in models:
        predictions = (model.predict(x_test))
        accuracy=metrics.accuracy_score(y_test,predictions)
        precision=metrics.precision_score(y_test,predictions)
        recall=metrics.recall_score(y_test,predictions)
        roc=roc_auc_score(y_test, predictions)
        scores[str(model)]=[accuracy,precision,recall, roc]
    scores.index=['accuracy','precision','recall','roc']
    return scores