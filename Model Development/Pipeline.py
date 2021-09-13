from Toolkit import *
import time
import argparse
import shap
import imblearn
from matplotlib import pyplot as plt
from Feature_Selection import main as fs
from Feature_Engineering import main as fe
from Imputation import main as imp
from Model import build as build_model
from Model import validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve, average_precision_score
from os import path, remove
from seaborn import swarmplot
from math import ceil


def load(*args):
    """
    Load all csv files based on arguments
    """
    return tuple(pd.read_csv(arg, delimiter=',', quotechar='"') for arg in args) if len(args) > 1 else pd.read_csv(args[0], delimiter=',', quotechar='"')


def combineall(*args):
    """
    combines (lab/vitals etc.) data from multiple sources
    """

    o_df = args[0].append([*args[1:]])

    # Replace nan values in one-hot encoded columns with zeroes
    cols = ['diagnosis_CORADS 4', 'diagnosis_CORADS 5', 'diagnosis_CORADS 6', 'diagnosis_Covid_code', 'diagnosis_Positieve PCR',
            'origin_Eigen woonomgeving', 'origin_Instelling (anders)', 'origin_Verpleeg-/verzorgingshuis', 'origin_GGZ instelling',
            'origin_Instelling voor revalidatie', 'origin_Instelling vr verpleging verz', 'origin_Overige instellingen']
    o_df[cols] = o_df[cols].fillna(0)
    return o_df


def combinekey(key, lv):
    """
    combines baseline lab+vitals data with key data
    """
    # Add boolean sex column
    key['female'] = key.apply(lambda row: 1 if row['sex'] == 'F' else 0, axis=1)

    # One hot encode some more cols
    diagnosis_dummies = pd.get_dummies(key['diagnosis'], prefix='diagnosis')
    origin_dummies = pd.get_dummies(key['admission_origin'], prefix='origin')
    key = pd.concat([key, diagnosis_dummies, origin_dummies], axis=1)

    # Add keyfile data
    key_cols = ['female', 'age'] + list(diagnosis_dummies.columns) + list(origin_dummies.columns)
    lv = lv[lv.index.isin(key.index)].join(key[key_cols])

    return lv


def combineloc(lab, vitals, on=("pseudo_id", "time_adm")):
    """
    combines data from a single location/hospital
    """
    return pd.merge(lab, vitals, how="outer", on=on)


def get_labels(key, meds):
    insign_patients = list(key[(key['hospitalisation_time'] <= args.insign_los) &
                               (~key['discharge_dest'].str.contains('ziekenhuis', case=False)) &
                               (~key['discharge_dest'].str.contains('hospice', case=False)) &
                               (key['group_discharged'] == 1) &
                               (~key.index.isin(meds[meds['description'].str.lower().str.startswith(
                                   ('remdesivir', 'dexamethason', 'tocilizumab'))]
                                                ['pseudo_id']))].index)

    sign_patients = list(key[~key.index.isin(insign_patients)].index)

    return insign_patients, sign_patients


def normalise(data):
    for col in data.columns:
        if len(data[col].unique()) == 1:
            data[col] = 0.0
        else:
            colmin = data[col].min()
            colmax = data[col].max()
            data.loc[:, col] = data.loc[:, col].apply(lambda row: (row - colmin) / (colmax - colmin))
    return data


def split_data(data, test_size, r_state):
    data_train, data_test = train_test_split(data, test_size=test_size, random_state=r_state)
    X_train, X_test = data_train[[i for i in data_train.columns if i != 'label']], data_test[[i for i in data_train.columns if i != 'label']]
    y_train, y_test = data_train['label'], data_test['label']

    return X_train, X_test, y_train, y_test


def swarm(yhat, x, y, palette, t, dest_path):
    """
    Generate and save a swarm plot
    :param yhat: Prediction probabilities Pandas DataFrame in long format
    :param x: labels column name
    :param y: probabilities column name
    :param palette: colour palette of the two classes
    :param t: threshold line value
    :param dest_path: output file path
    """
    print(yhat)
    ax = swarmplot(data=yhat, x=x, y=y, orient='v', palette=palette)
    ax.set(ylim=(0.0, 1.0))
    plt.axhline(y=t, color='r', linestyle='-')
    plt.xlabel('')
    ax.set_xticklabels(['Sign. Stay', 'Insign. Stay'])
    save_file(plt, dest_path, dpi=200)
    plt.clf()


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Walks through a pipeline that generates a machine learning model')

    # Turn 'nodes' in the flowchart on/off
    parser.add_argument("-fs", "--feat_sel", help="Turn on feature selection", required=False, action='store_true')
    parser.add_argument("-fe", "--feat_eng", help="Turn on feature engineering", required=False, action='store_true')
    parser.add_argument("-n", "--normalisation", help="Turn on normalisation", required=False, action='store_true')
    parser.add_argument("-i", "--imputation", help="Turn on imputation. This also generates a ML model.", required=False, action='store_true')
    parser.add_argument("--shap", help="Turn on SHAP plot.", required=False, action="store_true")

    # Specify file paths
    parser.add_argument("--haga_key", help="Path to Haga key file", required=False, default="/exports/reum/tverheijen/Haga_Data/02-02/Stats/Patient_Statistics_Haga.csv")
    parser.add_argument("--haga_lab", help="Path to Haga lab data", required=False, default="/exports/reum/tverheijen/Haga_Data/02-02/Processed_Data/lab_processed.csv")
    parser.add_argument("--haga_vitals", help="Path to Haga vitals data", required=False, default="/exports/reum/tverheijen/Haga_Data/02-02/Processed_Data/vitals_processed.csv")
    parser.add_argument("--haga_lab_isaric", help="Path to Haga baseline lab data", required=False,
                        default="/exports/reum/tverheijen/Haga_Data/02-02/Processed_Data/baseline_lab_Haga.csv")
    parser.add_argument("--haga_vitals_isaric", help="Path to Haga baseline vitals data", required=False,
                        default="/exports/reum/tverheijen/Haga_Data/02-02/Processed_Data/baseline_vitals_Haga.csv")
    parser.add_argument("--haga_meds", help="Path to Haga medication data", required=False, default="/exports/reum/tverheijen/Haga_Data/02-02/Processed_Data/medication_processed.csv")
    parser.add_argument("--lumc_key", help="Path to LUMC key file", required=False, default="/exports/reum/tverheijen/LUMC_Data/20210330/Stats/Patient_Statistics_LUMC.csv")
    parser.add_argument("--lumc_lab", help="Path to LUMC lab data", required=False, default="/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/lab_processed.csv")
    parser.add_argument("--lumc_vitals", help="Path to LUMC vitals data", required=False, default="/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/vitals_processed.csv")
    parser.add_argument("--lumc_lab_isaric", help="Path to LUMC baseline lab data", required=False,
                        default="/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/baseline_lab_LUMC.csv")
    parser.add_argument("--lumc_vitals_isaric", help="Path to LUMC baseline vitals data", required=False,
                        default="/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/baseline_vitals_LUMC.csv")
    parser.add_argument("--lumc_meds", help="Path to LUMC medication data", required=False, default="/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/medication_processed.csv")

    # Set parameters
    parser.add_argument("-d", "--data", help="Specify data to train on. When training on one of both clusters, testing is performed on other cluster.", required=False, choices=['b', 'h', 'l'], default='b')
    parser.add_argument("--self", help="Test model on the same centre it was built on.", required=False, action='store_true')
    parser.add_argument("-mc", "--missings_cutoff", help="Set the allowed fraction of missings per feature", required=False, type=float, default=0.33333)
    parser.add_argument("-l", "--insign_los", help="Set cutoff time in hours for an insignificant LOS", required=False, type=float, default=72.0)
    parser.add_argument("-v", "--test_size", help="Set test/validation set fraction size", required=False, type=float, default=0.3)
    parser.add_argument("-r", "--random_state", help="Set the random state (seed)", required=False, type=int, default=42)
    parser.add_argument("-m", "--max_iter", help="Set the max number of iterations for the iterative imputer", required=False, type=int, default=250)
    parser.add_argument("-s", "--initial_strategy", help="Set the initial imputation strategy for the iterative imputer", required=False,
                        default='mean', choices=['mean', 'median', 'most_frequent', 'constant'])
    parser.add_argument("-e", "--estimator", help="Set the estimator for the iterative imputer", required=False, default='BayesianRidge',
                        choices=['BayesianRidge', 'DecisionTreeRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor'])
    parser.add_argument("-o", "--optimiser", help="Set the score to optimise on", required=False, default='NPV', choices=['NPV', 'PPV', 'F1'])
    parser.add_argument("--prior", help="Set prior / class weights", required=False, action='store_true')
    parser.add_argument("--oversample", help="Turn on oversampling", required=False, action='store_true')
    parser.add_argument("--over_weight", help="Set the oversample fraction based on nr. of non-cases", required=False, type=float, default=0.5)
    parser.add_argument("-t", "--tolerance", help="Set the tolerance of the stopping condition", required=False, type=float, default=1e-3)
    parser.add_argument("--outer", help="Set the number of outer CV folds", required=False, type=int, default=6)
    parser.add_argument("--inner", help="Set the number of inner CV folds", required=False, type=int, default=3)
    parser.add_argument("--model", help="Specify model types to generate", required=False, nargs='+', default=['RF'])

    # Model parameters
    parser.add_argument("--n_estimators", help="Set the number of trees in the forest", required=False, type=lambda s: [int(item) for item in s.split(',')], default=[250])
    parser.add_argument("--max_features", help="The number of features to consider when looking for the best split", required=False, type=lambda s: [str(item) for item in s.split(',')], default=['sqrt'])
    parser.add_argument("--penalty", help="Specify the norm used in the penalization", required=False, type=lambda s: [str(item) for item in s.split(',')], default=['l1','l2','elasticnet'])
    parser.add_argument("-c", "--c", help="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization"
                        , required=False, type=lambda s: [float(item) for item in s.split(',')], default=[0.001,0.01,0.1,1.0])

    args = parser.parse_args()

    # Error check
    available_models = ['RF', 'Log']
    err = []
    for m in args.model:
        if m not in available_models:
            err.append(m)
    if len(err) > 0:
        ValueError(f"The following model{'s' if len(err) > 1 else ''} are unknown: {', '.join(err)}\nAvailable models: {available_models}")

    # Load Haga data
    if (args.data in ['h', 'b']) or (args.data == 'l' and not args.self):
        haga_meds = load(args.haga_meds)
        haga_key = exclude(args.haga_key, args.haga_key, treatment_limit=False, dis_other_hospital=False, symptom_onset=False)
        haga_key.set_index('pseudo_id', inplace=True)

        # Load preprocessed data if available, otherwise load and preprocess data
        if path.exists("Processed_Data/Baseline_Haga.csv"):
            haga = pd.read_csv("Processed_Data/Baseline_Haga.csv", index_col=0)
        else:
            haga_lab, haga_vitals, haga_lab_baseline, haga_vitals_baseline = load(args.haga_lab, args.haga_vitals, args.haga_lab_isaric, args.haga_vitals_isaric)

            # Combine lab and vitals data
            haga = combineloc(haga_lab, haga_vitals)

            haga_lab_baseline.set_index('pseudo_id', inplace=True)
            haga_vitals_baseline.set_index('pseudo_id', inplace=True)
            haga_baseline = pd.concat([haga_lab_baseline, haga_vitals_baseline], axis=1)

            # Get baseline data
            haga = get_baseline(haga[haga['pseudo_id'].isin(haga_key.index)], haga_key, [24 for _ in haga.columns[2:]])
            haga = haga_baseline[haga_baseline.index.isin(haga.index)].combine_first(haga)

            # Add keyfile data to lab + vitals data
            haga = combinekey(haga_key, haga)

            # Rename Haga's Corona Labscore 1
            haga.rename(columns={'Corona Labscore 1': 'Corona Labscore'}, inplace=True)

            # Change missing values for supp. oxygen to 0
            haga['Supplemental Oxygen (L/min)'].fillna(value=0, inplace=True)

            save_file(haga, "Processed_Data/Baseline_Haga.csv")

    # Load LUMC data
    if (args.data in ['l', 'b']) or (args.data == 'h' and not args.self):
        lumc_meds = load(args.lumc_meds)
        lumc_key = exclude(args.lumc_key, args.lumc_key, treatment_limit=False, dis_other_hospital=False, symptom_onset=False)
        lumc_key.set_index('pseudo_id', inplace=True)

        # Load preprocessed data if available, otherwise load and preprocess data
        if path.exists("Processed_Data/Baseline_LUMC.csv"):
            lumc = pd.read_csv("Processed_Data/Baseline_LUMC.csv", index_col=0)
        else:
            lumc_lab, lumc_vitals, lumc_lab_baseline, lumc_vitals_baseline = load(args.lumc_lab, args.lumc_vitals, args.lumc_lab_isaric, args.lumc_vitals_isaric)

            # Combine lab and vitals data
            lumc = combineloc(lumc_lab, lumc_vitals)

            lumc_lab_baseline.set_index('pseudo_id', inplace=True)
            lumc_vitals_baseline.set_index('pseudo_id', inplace=True)
            lumc_baseline = pd.concat([lumc_lab_baseline, lumc_vitals_baseline], axis=1)

            # Get baseline data
            lumc = get_baseline(lumc[lumc['pseudo_id'].isin(lumc_key.index)], lumc_key, [24 for _ in lumc.columns[2:]])
            lumc = lumc_baseline[lumc_baseline.index.isin(lumc.index)].combine_first(lumc)

            # Add keyfile data to lab + vitals data
            lumc = combinekey(lumc_key, lumc)

            # Change missing values for supp. oxygen to 0
            lumc['Supplemental Oxygen (L/min)'].fillna(value=0, inplace=True)

            save_file(lumc, "Processed_Data/Baseline_LUMC.csv")

    # Get list of patients in short and long group
    if args.self:
        if args.data == 'h':
            insign_patients, sign_patients = get_labels(haga_key, haga_meds)
            data = haga
        elif args.data == 'l':
            insign_patients, sign_patients = get_labels(lumc_key, lumc_meds)
            data = lumc
    else:
        iph, sph = get_labels(haga_key, haga_meds)
        ipl, spl = get_labels(lumc_key, lumc_meds)
        insign_patients, sign_patients = iph + ipl, sph + spl

        data = combineall(haga, lumc) if args.data == 'b' else haga if args.data == 'h' else lumc
        test = None if args.data == 'b' else haga if args.data == 'l' else lumc

    # Feature engineering
    if args.feat_eng:
        data = fe(data)
        if args.data != 'b' and not args.self:
            test = fe(test)

    # Feature selection
    if args.feat_sel:
        data = fs(data, insign_patients, sign_patients, args.missings_cutoff)
        save_file(data, f"Processed_Data/Baseline_{'Haga' if args.data in ['b', 'h'] else ''}{'LUMC' if args.data in ['b', 'l'] else ''}_SelectedFeatures.csv")
        if args.data != 'b' and not args.self:
            test = test[data.columns]

    # Normalisation
    if args.normalisation:
        data = normalise(data)
        save_file(data, f"Processed_Data/Baseline_{'Haga' if args.data in ['b', 'h'] else ''}{'LUMC' if args.data in ['b', 'l'] else ''}_Normalised.csv")

        if args.data != 'b' and not args.self:
            test = normalise(test)

    # Add labels
    data.loc[:, 'label'] = data.apply(lambda row: 1 if row.name in insign_patients else 0, axis=1)
    if args.data != 'b' and not args.self:
        test.loc[:, 'label'] = test.apply(lambda row: 1 if row.name in insign_patients else 0, axis=1)

    # Split dataset in train and test
    if args.data == 'b' or args.self:
        X_train, X_test, y_train, y_test = split_data(data, args.test_size, args.random_state)
    else:
        X_train, y_train = data[[i for i in data.columns if i != 'label']], data['label']
        X_test, y_test = test[[i for i in test.columns if i != 'label']], test['label']

    # Oversample to balance the dataset
    if args.oversample:
        X_train.reset_index(inplace=True)
        oversampler = imblearn.over_sampling.RandomOverSampler(sampling_strategy=0.5, random_state=args.random_state)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        X_train.set_index('index', inplace=True)

    print(f"Number of patients in train set: {len(X_train.index)}\nNumber of patients in test set: {len(X_test.index)}")

    # Imputation
    if args.imputation:
        X_train = imp(X_train, estimator=args.estimator, initial_strategy=args.initial_strategy, max_iter=args.max_iter, randomstate=args.random_state, tol=args.tolerance)
        X_test = imp(X_test, estimator=args.estimator, initial_strategy=args.initial_strategy, max_iter=args.max_iter,randomstate=args.random_state, tol=args.tolerance)
        save_file(X_train, f"Processed_Data/TRAIN_Baseline_{'Haga' if args.data in ['b', 'h'] else ''}{'LUMC' if args.data in ['b', 'l'] else ''}_Imputed_Normalised.csv")
        save_file(X_test, f"Processed_Data/TEST_Baseline_{'Haga' if args.data in ['b', 'h'] else ''}{'LUMC' if args.data in ['b', 'l'] else ''}_Imputed_Normalised.csv")

        # Load colours for plots
        clr_sign, clr_insign = colour_palette()[0], colour_palette()[-1]

        for i, m in enumerate(args.model):
            print("____________________")
            # Generate model
            if m == 'RF':
                model = build_model(X=X_train, y=y_train, m=m, n_outer=args.outer, n_inner=args.inner, optimise_on=args.optimiser,
                                    class_weights='balanced' if args.prior else None, r_state=args.random_state, n_estimators=args.n_estimators
                                    , max_features=args.max_features)
            elif m == 'Log':
                model = build_model(X=X_train.iloc[2:] if args.data == 'l' and not args.self else X_train,
                                    y=y_train.iloc[2:] if args.data == 'l' and not args.self else y_train
                                    , m=m, n_outer=args.outer, n_inner=args.inner, optimise_on=args.optimiser, class_weights='balanced' if args.prior else None
                                    , r_state=args.random_state, C=args.c, penalty=args.penalty, l1_ratio=[0.2 * i for i in range(1, 5)])

            # Run model on training set
            _, yhat = validate(model, X_train, y_train)
            yhat = pd.DataFrame(yhat)
            print("> Training y_hat:")
            print(y_train.value_counts())
            print(yhat)
            yhat['label'] = yhat.apply(lambda row: y_train[row.name], axis=1)
            yhat = yhat.rename(columns={1: 'Target Probability'})
            # Determine threshold
            yhat.sort_values(['label', 'Target Probability'], ascending=[False, False], inplace=True, ignore_index=True)
            if m == 'Log':
                # threshold makes sure 25% of cases would be correctly classified on training set
                t = yhat.loc[ceil(len(yhat[yhat['label'] == 1].index) * 0.25) - 1, 'Target Probability']
            elif m == 'RF':
                t = 0.5
            # Make a swarm plot on train set
            swarm(yhat=yhat, x='label', y='Target Probability', palette=[clr_sign, clr_insign], t=t,
                  dest_path=f"Predictions/{m}/SWARM_{m}_{args.data}{'_self' if args.self else ''}.jpg")

            scores, _ = validate(model, X_train, y_train, t=t)
            f1, auc, sensitivity, specificity, ppv, npv = scores['1']['f1-score'], scores['1']['ROCAUC'], \
                                                          scores['1']['recall'], scores['0']['recall'], \
                                                          scores['1']['precision'], scores['0']['precision']

            # Write results to txt file (delete old results first, if they exist)
            if path.exists(f"Predictions/{m}/Performance_{m}_{args.data}{'_self_' if args.self else '_'}{args.optimiser}.txt"):
                remove(f"Predictions/{m}/Performance_{m}_{args.data}{'_self_' if args.self else '_'}{args.optimiser}.txt")
            with open(f"Predictions/{m}/Performance_{m}_{args.data}{'_self_' if args.self else '_'}{args.optimiser}.txt", 'a+') as f:
                f.write("TRAINING PERFORMANCE")
                f.writelines([f"\n{['F1', 'ROC AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV'][i]}: {s}"
                              for i, s in enumerate([f1, auc, sensitivity, specificity, ppv, npv])])

            print(f">>>>>>>> Estimated {m} model performance:\nF1-score: {f1}\nROC AUC: {auc}\nSpecificty: {specificity}\nSensitivity: {sensitivity}"
                  f"\nPPV: {ppv}\nNPV: {npv}\n_________________________")

            # Run model on independent test set
            scores, yhat = validate(model, X_test, y_test, t=t)
            yhat = pd.DataFrame(yhat)
            yhat['label'] = yhat.apply(lambda row: y_test[row.name], axis=1)
            yhat = yhat.rename(columns={1: 'Target Probability'})
            # Make a swarm plot on the test set
            swarm(yhat=yhat, x='label', y='Target Probability', palette=[clr_sign, clr_insign], t=t,
                  dest_path=f"Predictions/{m}/TEST_SWARM_{m}_{args.data}{'_self' if args.self else ''}.jpg")

            f1, auc, sensitivity, specificity, ppv, npv = scores['1']['f1-score'], scores['1']['ROCAUC'], \
                                                          scores['1']['recall'], scores['0']['recall'], \
                                                          scores['1']['precision'], scores['0']['precision']

            with open(f"Predictions/{m}/Performance_{m}_{args.data}{'_self_' if args.self else '_'}{args.optimiser}.txt", 'a+') as f:
                f.write("\n__________________\nTEST PERFORMANCE")
                f.writelines([f"\n{['F1', 'ROC AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV'][i]}: {s}"
                              for i, s in enumerate([f1, auc, sensitivity, specificity, ppv, npv])])

            print(f">>>>>>>> Final {m} model performance:\nF1-score: {f1}\nROC AUC: {auc}\nSpecificty: {specificity}\nSensitivity: {sensitivity}"
                  f"\nPPV: {ppv}\nNPV: {npv}\nThreshold: {t}\n_________________________")

            # Plot PR curve
            avg_prec = round(average_precision_score(y_test, yhat['Target Probability']), 2)
            plot_precision_recall_curve(estimator=model, X=X_test, y=y_test, sample_weight=None, response_method='auto',
                                        name='', ax=None, pos_label=None, **{'label': f'Avg. Precision = {avg_prec}', 'color': colour_palette()[1]})
            leg = plt.legend(handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="upper right")
            save_file(plt, f"Predictions/{m}/PR_{m}_{args.data}{'_self' if args.self else ''}.jpg")
            plt.clf()

            # Plot ROC curve
            plot_roc_curve(model, X_test, y_test)
            save_file(plt, f"Predictions/{m}/ROC_{m}_{args.data}{'_self' if args.self else ''}.jpg")
            plt.clf()

            # Plot SHAP plot
            """
            explainer = shap.TreeExplainer(model) if m == 'RF' else shap.KernelExplainer(model)
            shap_values = explainer.shap_values(X_train)[1]
            shap.summary_plot(shap_values, X_train, X_train.columns, show=False)
            plt.tight_layout()
            save_file(plt, f"Predictions/{m}/SHAP_{m}_{args.data}{'_self' if args.self else ''}.jpg")
            plt.clf()
            """
            if m == 'RF':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)[1]
                shap.summary_plot(shap_values, X_test, X_test.columns, show=False)
            elif m == 'Log':
                from alibi.explainers import KernelShap
                print("explainer")
                explainer = KernelShap(model.predict_proba, link='logit')
                print("explainer.fit")
                explainer.fit(X_train, summarise_background=True)
                print("explaination")
                explanation = explainer.explain(X_test, l1_reg=False)
                print("summary plot")
                shap.summary_plot(explanation.shap_values[1], X_test, X_test.columns, show=False, max_display=10)
            plt.tight_layout()
            save_file(plt, f"Predictions/{m}/TEST_SHAP_{m}_{args.data}{'_self' if args.self else ''}.jpg")
            plt.clf()

    print(f"Runtime: {round((time.time() - start_time) / 60, 1)} minutes")
