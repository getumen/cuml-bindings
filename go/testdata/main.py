import os
import sys

import numpy as np
import pandas as pd
import treelite
import treelite_runtime
import xgboost as xgb
from sklearn import datasets, model_selection

if sys.platform == "win32" or sys.platform == "cygwin":
    shared_library_extension = "dll"
elif sys.platform == "darwin":
    shared_library_extension = "dylib"
else:
    shared_library_extension = "so"


seed = 42

breast_cancer = datasets.load_breast_cancer()

feature = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
target = pd.Series(breast_cancer.target)

train_x, test_x, train_y, test_y = model_selection.train_test_split(
    feature,
    target,
    test_size=0.2,
    shuffle=True,
    random_state=seed,
)

dtrain = xgb.DMatrix(train_x, label=train_y)

booster = xgb.train(
    {
        "max_depth": 6,
        "eta": 0.01,
        "objective": "binary:logistic",
        "seed": seed,
    },
    dtrain,
    100,
)

booster.save_model("xgboost.model")

test_x.to_csv("feature.csv", index=False, header=False, float_format="%.8f")
test_y.to_csv("label.csv", index=False,  header=False, float_format="%.8f")

dvalid = xgb.DMatrix(test_x)

xgboost_scores = booster.predict(dvalid)
with open("score-xgboost.csv", "w") as f:
    for x in xgboost_scores:
        print(x, file=f)

dvalid = treelite_runtime.DMatrix(test_x)

model = treelite.Model.from_xgboost(booster)

annotator = treelite.Annotator()
annotator.annotate_branch(model=model, dmat=dvalid, verbose=True)
annotator.save(path="annotation.json")

model.export_lib(
    toolchain="gcc",
    libpath=f"compiled-model.{shared_library_extension}",
    verbose=True,
    params={
        "parallel_comp": os.cpu_count(),
        "annotate_in": "annotation.json",
    },
)

predictor = treelite_runtime.Predictor(
    f"compiled-model.{shared_library_extension}",
    nthread=os.cpu_count(),
    verbose=True,
)

treelite_scores = predictor.predict(dvalid, verbose=True)
with open("score-treelite.csv", "w") as f:
    for x in treelite_scores:
        print(x, file=f)

np.testing.assert_array_almost_equal(xgboost_scores, treelite_scores, decimal=5)
