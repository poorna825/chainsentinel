"""Week 1 solutions. Open only after you've tried sections 7 and 8 yourself.
Paste each block into the matching notebook cell (it uses train, y_test and the sklearn imports from there).
"""
import numpy as np

# --- Section 7: random guesser ------------------------------------------
p = train["label"].mean()                       # illicit share in training data
rand_pred = (np.random.default_rng(42).random(len(y_test)) < p).astype(int)

print("accuracy :", round(accuracy_score(y_test, rand_pred), 3))
print("precision:", round(precision_score(y_test, rand_pred, zero_division=0), 3))
print("recall   :", round(recall_score(y_test, rand_pred), 3))
print("F1       :", round(f1_score(y_test, rand_pred), 3))
print(confusion_matrix(y_test, rand_pred, labels=[0, 1]))
# Recall ~= p and precision ~= the test illicit share: guessing gives no real signal.

# --- Section 8: flag everything ------------------------------------------
all_pred = np.ones(len(y_test), dtype=int)
print("accuracy :", round(accuracy_score(y_test, all_pred), 3))
print("precision:", round(precision_score(y_test, all_pred), 3))
print("recall   :", round(recall_score(y_test, all_pred), 3))   # 1.0 - catches all fraud...
print("F1       :", round(f1_score(y_test, all_pred), 3))
# ...but precision collapses: every normal user gets flagged. Recall alone can be gamed too.
# That is why we look at precision AND recall together (F1, PR-AUC).
