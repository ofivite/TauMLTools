import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

import os
import wandb
import neptune.new as neptune

random_state = 123
params = {
    "C":1.0,
    "test_size": 0.1,
}
wandb.init(config=params, project='test')
# run = neptune.init(project='yaourtpourtoi/test') # your credentials
# run["parameters"] = params

# Import some data to play with
X, y = datasets.make_moons(n_samples=10000, noise=0.3, random_state=random_state)
plt.scatter(X[:,0], X[:,1])
wandb.log({"scatter": plt})

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'],
                                                    random_state=random_state)

# Learn to predict each class against the other
model = LogisticRegression(penalty="l2", C=params['C'],
                                 random_state=random_state)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
wandb.log({"roc" : wandb.plot.roc_curve(y_test,
           y_proba, labels=None, classes_to_plot=[1])})
wandb.log({"precision-recall" : wandb.plot.pr_curve(y_test,
           y_proba, labels=None, classes_to_plot=[1])})
wandb.run.summary["accuracy"] = accuracy_score(y_test, y_proba[:, 1] > 0.5)
wandb.run.summary["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])

# custom roc curve
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=1)
data = list(zip(fpr, tpr))
table = wandb.Table(columns=["fpr", "tpr"], data=data)
roc_curve = wandb.plot_table(
    "wandb/area-under-curve/v0",
    table,
    {"x": "fpr", "y": "tpr"},
    {
        "title": "ROC",
        "x-axis-title": "False positive rate",
        "y-axis-title": "True positive rate",
    },
)
wandb.log({"custom_roc" : roc_curve})

# histogram with proba
data = [[s] for s in y_proba[:, 1]]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({'my_histogram': wandb.plot.histogram(table, "scores",
                           title="Histogram")})
# wandb.log({'roc': plt})
# run['interactive_img'].upload(neptune.types.File.as_html(plotly_fig))
