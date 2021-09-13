import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, precision_score, f1_score, make_scorer
from statistics import mean, stdev

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


def validate(m, X_test, y_test, t=0.5):
	"""
	Validates model on unseen data
	:param m: model object
	:param X_test: test dataframe
	:param y_test: labels
	:param t: probability threshold. 0.0 classifies all as 0, 1.0 classifies all as 1
	:return: nested dictionary with two main dictionaries ('0' and '1') for each class, holding 'f1-score', 'precision', 'recall', 'support', and 'ROCAUC'
			 see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
			 yhat, prediction probabilities for the two classes
	"""

	yhat = m.predict_proba(X_test)
	y_pred = [0 if i < t else 1 for i in yhat[:, 1]]

	if t == 0.5:
		print(pd.DataFrame(y_pred).iloc[:, 0].value_counts())

	roc_auc = roc_auc_score(y_test, yhat[:, 1])
	scores = classification_report(y_test, y_pred, output_dict=True)
	scores['0']['ROCAUC'], scores['1']['ROCAUC'] = roc_auc, roc_auc

	return scores, yhat


def build(X, y, m, n_outer, n_inner, optimise_on, class_weights, r_state, **kwargs):
	"""
	:param X: full training set
	:param y: full training set's labels
	:param m: model type ['RF', 'Log']
	:param n_outer: number of outer CV folds
	:param n_inner: number of inner CV folds
	:param optimise_on: what scoring metric to optimise on. Options: F1, NPV, PPV
	:param class_weights: class weights / prior probability. Either '' or 'Balanced'
	:param r_state: random state
	:param kwargs: model hyperparameters. When passed as a list, values will be added to the grid search for hyperparameter tuning
	:return: models of outer loop, dictionary with F1 and ROC AUC scores for each loop
	"""

	def scorer(y_test, y_pred, score=optimise_on):
		if score == 'NPV':
			return precision_score(y_true=y_test, y_pred=y_pred, pos_label=0)
		elif score == 'PPV':
			return precision_score(y_true=y_test, y_pred=y_pred, pos_label=1)
		elif score == 'F1':
			return f1_score(y_true=y_test, y_pred=y_pred, pos_label=1)

	# Replace pseudo_id indexes for integers
	X.reset_index(drop=True, inplace=True)
	y.reset_index(drop=True, inplace=True)

	# Set up outer CV class
	cv_outer = KFold(n_splits=n_outer, shuffle=True, random_state=r_state)
	ensemble = []
	outer_results = {'F1': [], 'ROCAUC': [], 'spec': [], 'sens': [], 'PPV': [], 'NPV': []}
	outer_i = 0

	for train_rows, test_rows in cv_outer.split(X):
		# Split data (into training folds and test fold)
		X_train, X_test = X.loc[train_rows, :], X.loc[test_rows, :]
		y_train, y_test = y[train_rows], y[test_rows]

		# Set up the inner CV class
		cv_inner = KFold(n_splits=n_inner, shuffle=True, random_state=r_state)

		# Initialise the model class with restricted hyperparameters (which won't get tuned during CV)
		if m == 'RF':
			model = RandomForestClassifier(class_weight=class_weights, random_state=r_state, **{i: kwargs[i][0] for i in kwargs if len(kwargs[i]) == 1})
		if m == 'Log':
			model = LogisticRegression(class_weight=class_weights, random_state=r_state, solver='saga', max_iter=1000, **{i: kwargs[i][0] for i in kwargs if len(kwargs[i]) == 1})

		# Set up the search space (tunable hyperparameters)
		space = {p: v for p, v in kwargs.items() if len(v) > 1}

		# Define search with custom NPV scoring function
		custom_scorer = make_scorer(scorer, greater_is_better=True)
		search = GridSearchCV(model, space, scoring=custom_scorer, cv=cv_inner)
		# Execute search
		result = search.fit(X_train, y_train)
		best_model = result.best_estimator_
		# Train best inner loop model on outer loop training set
		best_model.fit(X_train, y_train)
		# Run best inner loop model on outer loop iteration test set
		scores, _ = validate(best_model, X_test, y_test)

		# Save model and model results
		ensemble.append(best_model)
		outer_results['F1'].append(scores['1']['f1-score'])
		outer_results['ROCAUC'].append(scores['1']['ROCAUC'])
		outer_results['sens'].append(scores['1']['recall'])
		outer_results['spec'].append(scores['0']['recall'])
		outer_results['PPV'].append(scores['1']['precision'])
		outer_results['NPV'].append(scores['0']['precision'])

		# Report progress
		print(f'PPV={outer_results["PPV"][-1]}, NPV={outer_results["NPV"][-1]}, Specificity={outer_results["spec"][-1]}, cfg={result.best_params_}, '
			  f'train cases={np.count_nonzero(y_train == 1)}, test cases={np.count_nonzero(y_test == 1)}')

		outer_i += 1

	# Summarize the mean performance of the ensemble
	f1, auc, sensitivity, specificity, ppv, npv = mean(outer_results['F1']), mean(outer_results['ROCAUC']), \
												  mean(outer_results['sens']), mean(outer_results['spec']), \
												  mean(outer_results['PPV']), mean(outer_results['NPV'])
	print(f">>>>>>>> Mean {m} ensemble performance:\nF1-score: {f1}\nROC AUC: {auc}\nSpecificty: {specificity}\nSensitivity: {sensitivity}"
		  f"\nPPV: {ppv}\nNPV: {npv}\n_________________________")

	# Choose the best model based on the set optimise_on score (default=NPV)
	best_model = ensemble[outer_results[optimise_on].index(max(outer_results[optimise_on]))]

	# Make new model on entire train set, with hyperparams tuned identically to best model
	params = best_model.get_params()
	model.set_params(**params)
	model.fit(X, y)

	return model
