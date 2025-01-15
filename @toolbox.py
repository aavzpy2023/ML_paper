import json
import os
from collections import Counter
from datetime import timedelta
from itertools import repeat
from warnings import filterwarnings

import joblib
import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.preprocessing import RobustScaler

np
filterwarnings("ignore", category=DeprecationWarning)
models = {
    1: "bag_clf",
    2: "pas_clf",
    3: "adaboost",
    4: "bag_clf-pas_clf",
    5: "bag_clf-adaboost",
    6: "pas_clf-adaboost",
    7: "bag_clf-past_clf-adaboost",
    8: "rf",
}

k_data = {1: "relabeled", 2: "original"}
months = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
init_month = 1
data_x_mz = pd.read_excel(
    "todo_ok.xlsx",
    sheet_name="Datos por MZ",
    usecols=["MZ", "Norte", "Sur", "Oeste", "Este"],
    dtype={
        "MZ": str,
        "Norte": str,
        "Sur": str,
        "Oeste": str,
        "Este": str,
    },
)
scaler = RobustScaler()
result_to_txt = list()
width_of_text = 280


def select_model() -> str:
    """_summary_
    Get the model to be used
    Returns: Name of the model to be used
        _type_: _description_
    """
    model_id = ""
    while not model_id.isdigit():
        model_id = input(
            "The following models can be selected to train and test it:\n"
            r"\n1 Bagging Classifier with Random Forest (bag_clf)"
            r"\n2 Pasting Classifier with Random Forest (pas_clf)"
            r"\n3 AdaBoost Classifier with Random Forest (AdaBoost)"
            r"\n4 bag_clf - pas_clf"
            "\n5 bag_clf - AdaBoost"
            r"\n6 pas_clf - AdaBoost"
            "\n7 bag_clf - pas_clf - AdaBoost"
            r"\n\nPlease type the number of the model:"
        )
        if model_id.isdigit():
            if int(model_id) > 7 or int(model_id) < 1:
                model_id = ""

    print("The selected model is:", models.get(int(model_id)))
    res = str(models.get(int(model_id)))
    return res


def select_data() -> str:
    """_summary_
    Get the kind of data to be used
    Returns: Kind the data to be used
        _type_: _description_
    """
    kind_data_set = ""
    while not kind_data_set.isdigit():
        kind_data_set = input(
            "There are two scenarios for the data:\n\n"
            "1 Data of blocks identified as negative will\
                                    be relabeled\n"
            "2 Original data (no data will be relabeled)\n"
            "\n\nPlease type the number of the data to be\
                                    used:"
        )
        if kind_data_set.isdigit():
            if int(kind_data_set) < 1 or int(kind_data_set) > 2:
                kind_data_set = ""
    return k_data.get(int(kind_data_set))


def process_data(k_data: str) -> pd.DataFrame:
    """_summary_
    Process the data for the model
    Args:
        k_data (str): kind of data to be used
    Returns: DataFrame ready to be used
        pd.DataFrame: _description_
    """
    csv_data = pd.read_csv("ok_data.csv", parse_dates=[2], index_col=0)
    # Relabel data if necessary
    if k_data != "relabeled":
        full_data = csv_data.copy()
    else:
        full_data = relabel_data(csv_data)
    return full_data


def relabel_data(pd_data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Relabel all blocks identified as negative
    Args:
        pd_data (pd.DataFrame): DataFrame with all data
    Returns:
        pd.DataFrame: _description_
    """
    print("Relabeling all blocks identified as negative")
    init_date = pd.to_datetime("2019-03-01", format="%Y-%m-%d")
    full_data = pd_data.copy()
    full_data["date"] = pd.to_datetime(full_data["date"], format="mixed")
    full_data_cluster = full_data[full_data.date >= init_date]
    data_to_cluster = full_data_cluster[(full_data_cluster["status"] == 0)].loc[
        :, "1_week_ago":"8_week_ago"
    ]
    km = KMeans(n_clusters=2, random_state=0)
    print(
        "A total of",
        full_data_cluster[full_data_cluster["status"] == 1].shape[0],
        "are classified as positive (1)",
    )
    print(data_to_cluster.shape[0], "items will be relabeled")
    km.fit(scaler.fit_transform(data_to_cluster), full_data_cluster.status)
    neg = full_data_cluster[full_data_cluster["status"] == 0]
    neg.insert(neg.shape[1], "status_clf", km.labels_, True)
    pos = full_data_cluster[full_data_cluster["status"] == 1]
    pos.insert(pos.shape[1], "status_clf", np.ones(pos.shape[0]).reshape((-1, 1)), True)
    full_data = pd.concat([neg, pos])
    print("All blocks identified as negative has been relabeled")
    positives_after_relabel = full_data[full_data["status_clf"] == 1].shape[0]
    print(f"Now, a total of {positives_after_relabel} are positive (1)")
    full_data = full_data.drop("status", axis=1)
    full_data.columns = [
        "MZ",
        "date",
        "week",
        "1_week_ago",
        "2_week_ago",
        "3_week_ago",
        "4_week_ago",
        "5_week_ago",
        "6_week_ago",
        "7_week_ago",
        "8_week_ago",
        "status",
    ]
    return full_data


def get_estimators_table() -> tuple:
    """_summary_
    Get estimators' values
    Returns:
        tuple: tuple of values of each estimators
    """
    estimators_table = pd.read_csv("best_estimators.csv", index_col=0)
    rf_estimator = estimators_table.best_data_original["rf"]
    bag_estimator = estimators_table.best_data_original["bag_clf"]
    pas_estimator = estimators_table.best_data_original["bag_clf"]
    ada_estimator = estimators_table.best_data_original["adaboost"]
    return rf_estimator, bag_estimator, pas_estimator, ada_estimator, estimators_table


def train_model(
    model_name: str, full_data: pd.DataFrame, k_data: str, train_test=False
) -> StackingClassifier:
    """_summary_
        Train selected model
    Args:
        model_name (str): name of model to train
        full_data (pd.DataFrame): Dataframe with all data
        k_data (str): kind of data for training (original or relabeled all
                                                 block identified as negative)
        train_test (bool): True if the training set will be used for both
        training and testing
    Returns:
        StackingClassifier: _description_
    """
    voting_clf = select_model_to_run(model_name, k_data)
    print_sms("The parameters for each model was taken from table.")
    print_sms(f"starting training of model '{model_name}' at ", Timestamp.now())
    first_date = pd.to_datetime("2019-03-01", format="mixed")
    predict_date_begin = pd.to_datetime("2022-1-1", format="mixed")
    full_data["date"] = pd.to_datetime(full_data["date"], format="mixed")
    training_data = full_data[
        (full_data["date"] > first_date) & (full_data["date"] < predict_date_begin)
    ]
    print_sms(f"Percentage of training set: 70")
    print_sms(f"Percentage of testing set: 30")
    X = training_data.loc[:, "1_week_ago":"8_week_ago"]
    # status col was deleted and status_clf renamed for status
    y = training_data["status"]
    print("aaaaaaa    ", train_test)
    if train_test:
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
        voting_clf.fit(X_train, y_train)
        print_sms("finished training at ", Timestamp.now())
        y_predict = voting_clf.predict(X_test)
        cf = confusion_matrix(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        prec = precision_score(y_test, y_predict)
        f1_sc = f1_score(y_test, y_predict)
        print(
            f"{model_name}__precision:{round(prec, 4)}\
                __recall:{round(recall, 4)}__f1_score:{round(f1_sc, 4)}"
        )
        print(cf)
    else:
        X_train = scaler.fit_transform(X)
        voting_clf.fit(X_train, y)
        print_sms("finished training at ", Timestamp.now())
    return voting_clf


def select_model_to_run(model_name: str, k_data: str):
    """
    Select the model to run
    model_name (str): name of model to train
    k_data (str): kind of data for training (original or relabeled all block identified as negative)

    """
    est_table = get_estimators_table()[-1]
    if k_data == "relabeled":
        rf_estimator, bag_estimator, pas_estimator, ada_estimator, bg_pg_ada = (
            est_table.best_data_labeled
        )
        rf_max_leaf, bag_max_leaf, pas_max_leaf, ada_max_leaf, bg_pg_ada = (
            est_table.max_leaf_data_labeled
        )
    else:
        rf_estimator, bag_estimator, pas_estimator, ada_estimator, bg_pg_ada = (
            est_table.best_data_original
        )
        rf_max_leaf, bag_max_leaf, pas_max_leaf, ada_max_leaf, bg_pg_ada = (
            est_table.max_leaf_data_original
        )
    raf_clf = RandomForestClassifier(
        n_estimators=int(rf_estimator), random_state=42, max_features=1.0, n_jobs=-1
    )
    print_sms("Training model")
    if model_name == "bag_clf-past_clf-adaboost":
        print("running bag_clf-past_clf-adaboost model")
        ada_clf = AdaBoostClassifier(
            raf_clf,
            n_estimators=int(ada_estimator),
            algorithm="SAMME.R",
            learning_rate=0.05,
        )
        bag_clf = BaggingClassifier(
            RandomForestClassifier(),
            n_estimators=int(bag_estimator),
            bootstrap=True,
            n_jobs=-1,
        )
        pas_clf = BaggingClassifier(
            RandomForestClassifier(),
            n_estimators=int(pas_estimator),
            bootstrap=False,
            n_jobs=-1,
        )
        voting_clf = StackingClassifier(
            estimators=[("bag", bag_clf), ("pas", pas_clf), ("ada", ada_clf)],
            final_estimator=RandomForestClassifier(random_state=43, n_jobs=-1),
        )
    elif model_name == "pas_clf-adaboost":
        print("running pas_clf-adaboost model")
        ada_clf = AdaBoostClassifier(
            raf_clf,
            n_estimators=int(ada_estimator),
            algorithm="SAMME.R",
            learning_rate=0.05,
        )
        pas_clf = BaggingClassifier(
            RandomForestClassifier(),
            n_estimators=int(pas_estimator),
            bootstrap=False,
            n_jobs=-1,
        )
        voting_clf = StackingClassifier(
            [("pas", pas_clf), ("ada", ada_clf)],
            final_estimator=RandomForestClassifier(random_state=43, n_jobs=-1),
        )
    elif model_name == "bag_clf-adaboost":
        print("running bag_clf-adaboost model")
        ada_clf = AdaBoostClassifier(
            raf_clf,
            n_estimators=int(ada_estimator),
            algorithm="SAMME.R",
            learning_rate=0.05,
        )
        bag_clf = BaggingClassifier(
            RandomForestClassifier(),
            n_estimators=int(bag_estimator),
            bootstrap=True,
            n_jobs=-1,
        )
        voting_clf = StackingClassifier(
            [("bag", bag_clf), ("ada", ada_clf)],
            final_estimator=RandomForestClassifier(random_state=43, n_jobs=-1),
        )
    elif model_name == "bag_clf-pas_clf":
        print("running bag_clf-pas_clf model")
        bag_clf = BaggingClassifier(
            RandomForestClassifier(),
            n_estimators=int(bag_estimator),
            bootstrap=True,
            n_jobs=-1,
        )
        pas_clf = BaggingClassifier(
            RandomForestClassifier(),
            n_estimators=int(pas_estimator),
            bootstrap=False,
            n_jobs=-1,
        )
        voting_clf = StackingClassifier(
            [("bag", bag_clf), ("pas", pas_clf)],
            final_estimator=RandomForestClassifier(random_state=43),
            n_jobs=-1,
        )
    elif model_name == "adaboost":
        print("running adaboost model")
        raf_clf = RandomForestClassifier(
            max_leaf_nodes=int(ada_max_leaf),
            n_estimators=int(rf_estimator),
            random_state=42,
            n_jobs=-1,
        )
        voting_clf = AdaBoostClassifier(
            raf_clf,
            n_estimators=int(ada_estimator),
            algorithm="SAMME.R",
            learning_rate=0.05,
        )
    elif model_name == "pas_clf":
        print("running pas_clf model")
        raf_clf = RandomForestClassifier(
            max_leaf_nodes=int(pas_max_leaf),
            n_estimators=int(rf_estimator),
            random_state=42,
            n_jobs=-1,
        )
        voting_clf = BaggingClassifier(
            RandomForestClassifier(),
            n_estimators=int(pas_estimator),
            bootstrap=False,
            n_jobs=-1,
        )
    else:  # bag_clf
        print("running bag_clf model")
        raf_clf = RandomForestClassifier(
            max_leaf_nodes=int(bag_max_leaf),
            n_estimators=int(rf_estimator),
            random_state=42,
            n_jobs=-1,
        )
        voting_clf = BaggingClassifier(
            RandomForestClassifier(),
            n_estimators=int(bag_estimator),
            bootstrap=True,
            n_jobs=-1,
        )
    return voting_clf


def test_model(voting_clf: StackingClassifier, full_data: pd.DataFrame) -> tuple:
    """_summary_
    Test the model in the testing set
    Args:
        voting_clf (StackingClassifier): StackingClassifier object
        full_data (pd.DataFrame): Dataframe with all data

    Returns:
        tuple: _description_
    """
    print_sms("Testing model")
    # scaler = MinMaxScaler(feature_range=(0, 1))
    true_negative = list()
    false_negative = list()
    true_positive = list()
    false_positive = list()
    all_precisions = list()
    all_recall = list()
    all_f1score = list()
    all_prediction_2022 = dict()
    from datetime import timedelta

    months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    init_month = 1
    all_pred_data_with_prediction = list()
    for i in range(init_month, 13):
        print(f"Testing the model on {months[i-1]} 2022")
        tmp_predict_date_begin = pd.to_datetime(f"{i}/1/2022", format="%m/%d/%Y")
        next_month_day = tmp_predict_date_begin + timedelta(days=31)
        tmp_predict_date_end = next_month_day - timedelta(days=next_month_day.day - 1)
        X_predict_data = full_data[
            (full_data["date"] >= tmp_predict_date_begin)
            & (full_data["date"] < tmp_predict_date_end)
        ]
        X_predict = X_predict_data.loc[:, "1_week_ago":"8_week_ago"]
        input_data = scaler.fit_transform(X_predict)
        # print('-----PREDICTING---------')
        vt_clf_prediction = voting_clf.predict(input_data)
        prec_score = precision_score(X_predict_data.status, vt_clf_prediction)
        rec_score = recall_score(X_predict_data.status, vt_clf_prediction)
        f1_sco = f1_score(X_predict_data.status, vt_clf_prediction)
        # print(f'--Predicion in {months[i-1]}: precision: {round(prec_score, 4)}__recall:{round(rec_score, 4)}__f1:{round(f1_sco, 4)}')
        X_predict_data.insert(
            X_predict_data.shape[1], "prediction", vt_clf_prediction, True
        )
        # all_pred_data_with_prediction[months[i-1]] = X_predict_data
        all_prediction_2022[months[i - 1]] = tuple(
            X_predict_data[X_predict_data["prediction"] == 1].MZ
        )
        cf_mt = confusion_matrix(X_predict_data.status, vt_clf_prediction)
        true_negative.append(cf_mt[0, 0])
        false_positive.append(cf_mt[0, 1])
        true_positive.append(cf_mt[1, 1])
        false_negative.append(cf_mt[1, 0])
        all_precisions.append(prec_score)
        all_recall.append(rec_score)
        all_f1score.append(f1_sco)
        voting_clf.fit(scaler.fit_transform(X_predict), X_predict_data.status)
    all_results = pd.DataFrame(
        {
            "False_neg": false_negative,
            "True_neg": true_negative,
            "False_pos": false_positive,
            "True_pos": true_positive,
            "PR": all_precisions,
            "RC": all_recall,
            "F1": all_f1score,
        }
    )
    all_results.index = months[init_month - 1 :]
    print_sms("Testing finished at", Timestamp.now())
    return all_results, all_prediction_2022  # , all_pred_data_with_prediction


def get_besides_zones(zone: str) -> dict:
    """
    Get the zones beside any given zone
    Args:
        zone: string with the number of the zone

    Returns:
        Dict: with the number on zone by cardinal points
    """
    beside = data_x_mz[data_x_mz["MZ"] == str(zone)]
    results = dict()
    results["Norte"] = __separate_beside_zones(beside["Norte"].iloc[0])
    results["Sur"] = __separate_beside_zones(beside["Sur"].iloc[0])
    results["Este"] = __separate_beside_zones(beside["Este"].iloc[0])
    results["Oeste"] = __separate_beside_zones(beside["Oeste"].iloc[0])
    return results


def __separate_beside_zones(zone: str) -> tuple:
    """
    Split beside zone

    Args:
        zone: any string containing the data

    Retruns:
        tuple: all number of the zones
    """
    zone_str = zone
    if type(zone) is not str:
        zone_str = str(zone)
    split = (zone_str,)
    if len(zone_str) > 1:
        if "," in zone_str:
            split = zone_str.split(",")
    return tuple(split)


def check_arround_zone_is_positives_future(
    besid_zone: str, first_week_pos_zone: tuple, pos_zone_after_1st_week: tuple
):
    """_summary_
    Check if any zone arround of selected zone is positive in next days of month
    Args:
        besid_zone (str): name of zone
        first_week_pos_zone (tuple): positive zones in first week
        pos_zone_after_1st_week (tuple): positives zone after first week

    Returns:
        _type_: Name of zone
    """
    if besid_zone not in first_week_pos_zone and besid_zone in pos_zone_after_1st_week:
        return besid_zone


def get_model_performance(
    all_prediction_2022: dict, full_data: pd.DataFrame, all_results: pd.DataFrame
) -> tuple:
    """_summary_
    Args:
        all_prediction_2022 (dict): Prediction of every month
        full_data (pd.DataFrame): full dataset
        all_results (pd.DataFrame): results of the model's testing

    Returns:
        tuple: (result by month, performance by month)
    """
    all_pred_keys = tuple(all_prediction_2022.keys())
    total_zones = len(Counter(full_data.MZ).keys())
    # len(all_prediction_2022.get(all_pred_keys[0]))
    pos_per_1st_week = list()
    total_prediction_month_match = []
    real_month = []
    accuracy_per_month = []
    total_pos_model_month = list()
    pos_zone_future = list()
    len_pos = 0
    pos_zon_not_inspected_on_time = list()
    mz_pos_por_no_tratar_mz_adyacentes_previamente = list()
    pos_model_neg_real = list()
    months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    all_prec = list()
    all_rec = list()
    all_f1 = list()
    for i in range(init_month, 13):
        date_begin = pd.to_datetime(f"{i}/1/2022", format="%m/%d/%Y")
        next_month_day = date_begin + timedelta(days=31)
        date_end = next_month_day - timedelta(days=next_month_day.day - 1)
        # Select just the data positive
        positive_zone_in_period = full_data[
            (full_data.date >= date_begin)
            & (full_data.date < date_end)
            & (full_data.status == 1)
        ]
        negative_zone_in_period = full_data[
            (full_data.date >= date_begin)
            & (full_data.date < date_end)
            & (full_data.status == 0)
        ]
        prec_score = 0
        rec_score = 0
        f1 = 0
        if positive_zone_in_period.shape[0] > 1:
            # total positive zones predicted by model
            prediction_of_the_month = Counter(
                all_prediction_2022.get(months[i - 1])
            ).keys()
            total_pos_model_month.append(len(prediction_of_the_month))
            zones_matches = set(prediction_of_the_month).intersection(
                set(positive_zone_in_period.MZ)
            )  # True positive
            zones_matches_FP = set(negative_zone_in_period.MZ).intersection(
                set(prediction_of_the_month)
            )
            all_zones_name = set(Counter(process_data("original").MZ).keys())
            total_prediction_month_match.append(len(zones_matches))
            real_month.append(positive_zone_in_period.shape[0])
            acc_month = len(zones_matches) / positive_zone_in_period.shape[0]
            accuracy_per_month.append(round(acc_month, 4))
            # days_to_inspect_predicted_zones (dipz)
            dipz = round(
                len(prediction_of_the_month) / 9
            )  # AVG: 9 blocks inspected per day
            pos_zones_first_week = set(
                positive_zone_in_period[
                    (positive_zone_in_period.date >= date_begin)
                    & (
                        positive_zone_in_period.date
                        <= date_begin + timedelta(days=dipz)
                    )
                ].MZ
            )
            # Zones not inspected in first week according prediction (znifw)
            zones_not_detected = len(zones_matches.difference(pos_zones_first_week))
            pos_zon_not_inspected_on_time.append(zones_not_detected)
            # Positives zones due to posivites zones predicted by model not treated on time (pz)
            dif_pred_1st_week = set(zones_matches).difference(pos_zones_first_week)
            for zone in dif_pred_1st_week:
                if len(str(zone)) == 1:
                    bes_zones = tuple(get_besides_zones(f"0{zone}").values())
                else:
                    bes_zones = tuple(get_besides_zones(zone).values())
                ok_bes_zones = []
                tmp = []
                for card_point in bes_zones:
                    if len(card_point) == 1:
                        if card_point[0] != "-":
                            tmp.append(card_point[0])
                    else:
                        for it in card_point:
                            if card_point[0] != "-":
                                tmp.append(it)
                ok_bes_zones.extend(tmp)
                all_pos_zones_first_week = tuple(
                    [pos_zones_first_week for _ in range(len(ok_bes_zones))]
                )
                all_dif_pred_1st_week = tuple(
                    [dif_pred_1st_week for _ in range(len(ok_bes_zones))]
                )
                zone_in_future = list(
                    map(
                        check_arround_zone_is_positives_future,
                        ok_bes_zones,
                        all_pos_zones_first_week,
                        all_dif_pred_1st_week,
                    )
                )
                if len(zone_in_future) > 1:
                    for k in zone_in_future:
                        if k not in (None, "-") or type(k) in (list, tuple, str):
                            if type(k) is str:
                                pos_zone_future.append(k)
                            else:
                                for q in k:
                                    if type(q) in (list, tuple):
                                        pos_zone_future.extend(q)
                                    elif type(q) is str:
                                        pos_zone_future.append(q)
                    if len(pos_zone_future) > 1:
                        pos_zone_future = list(Counter(pos_zone_future).keys())
                    if "-" in pos_zone_future:
                        pos_zone_future.remove("-")
                elif type(zone_in_future) is str:
                    pos_zone_future.append(zone_in_future)
            if len(pos_zone_future) > len_pos:
                dif = len(pos_zone_future) - len_pos
                mz_pos_por_no_tratar_mz_adyacentes_previamente.append(dif)
            else:
                mz_pos_por_no_tratar_mz_adyacentes_previamente.append(0)
            len_pos = len(pos_zone_future)
            # Total of zones predicted as positives and was negative in real data (PvN)
            dif_pos_model_vs_neg_real = set(prediction_of_the_month).difference(
                set(positive_zone_in_period.MZ)
            )
            pos_model_neg_real.append(len(dif_pos_model_vs_neg_real))
    all_res = pd.DataFrame(
        {
            "Zones_pred_model": total_pos_model_month,
            "zones_detected": real_month,
            "Zones_pred_model_match": all_results["True_pos"],
            # 'model_accuracy': np.array(accuracy_per_month),
            "precision": all_results.PR,
            "recall": all_results.RC,
            "f1_score": all_results["F1"],
            # zones inspected on time (zion)
            "zion": np.array(total_prediction_month_match)
            - np.array(
                pos_zon_not_inspected_on_time
            ),  #'match_1s_week': posi_first_week_match
            "znifw": pos_zon_not_inspected_on_time,
            "pz": mz_pos_por_no_tratar_mz_adyacentes_previamente,
            "PvN": all_results["False_pos"],
        }
    )
    all_res.index = months[init_month - 1 :]
    print_sms(all_res.to_string())
    # print_sms(all_res.sum())
    # print_sms(all_res.mean())
    report = all_results.copy()
    months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    report.index = months[init_month - 1 :]
    good_classification = report.True_neg + report.True_pos
    total_classification = report.sum(axis=1)
    percentage = good_classification / total_classification
    report["tot_pron_corr"] = good_classification
    report["total_pron"] = total_classification
    report["percent_pron_corr"] = percentage
    print_sms("ACCURACY OF THE MODEL:", np.average(report.percent_pron_corr))
    print_sms(report.to_string())
    return all_res, report


def run_tool_box() -> tuple:
    """_summary_
    Get the choice of models to be used
    Returns:
        int: Number of choice selected
    """
    print_sms("Please, select one of the following options:")
    print_sms("1 Find best estimators (by GridSearchCV) for models(4 models)")
    print_sms("2 Train and test model (8 models)")
    print_sms("3 Predict (with any of saved models trained previously)")
    print_sms("4 Exit")
    action = ""

    while not action.isdigit():
        action = input("Type here the selected option:")
        if action.isdigit():
            if int(action) < 1 or int(action) > 4:
                action = ""
    selected_action = int(action)
    print("".ljust(width_of_text, "-"))
    if selected_action == 4:
        exit()
    elif selected_action == 1:
        print_sms("The following models can be used:")
        print_sms("1 Bagging Classifier with Random Forest (bag_clf)")
        print_sms("2 Pasting Classifier with Random Forest (pas_clf)")
        print_sms("3 AdaBoost Classifier with Random Forest (AdaBoost)")
        print_sms("4 Random Forest (rf)")
        print_sms("With two different dataset")
        print_sms("a blocks identified as negative will be relabeled")
        print_sms("b original (no relabeled)")
    else:
        print_sms("The following models can be used:")
        print_sms("1 Bagging Classifier with Random Forest (bag_clf)")
        print_sms("2 Pasting Classifier with Random Forest (pas_clf)")
        print_sms("3 AdaBoost Classifier with Random Forest (AdaBoost)")
        print_sms("4 bag_clf - pas_clf")
        print_sms("5 bag_clf - AdaBoost")
        print_sms("6 pas_clf - AdaBoost")
        print_sms("7 bag_clf - pas_clf - AdaBoost")
        print_sms("8 Random Forest")
        print_sms("With two different dataset")
        print_sms("a blocks identified as negative will be relabeled")
        print_sms("b original (no relabeled)")
    models_to_train = ""
    incorrect_models, correct_models = repeat(list(), 2)
    while len(incorrect_models) > 0 or len(correct_models) == 0:
        if selected_action == 3:
            print_sms(
                "Please type the combination of the model and kind of data (e.g. 1a)"
            )
            models_to_train = input("Type here the model combination:")
        else:
            print_sms("Please type the combination of the model(s) and kind of data")
            print_sms("For only one model (e.g: 1a)")
            print_sms("For more than one model (e.g: 1a,2B,3a)")
            print_sms("All models (e.g: all)")
            models_to_train = input("Type here the model(s):")
        incorrect_models, correct_models = check_models(
            models_to_train, selected_action
        )
    print(
        f"A total of {len(correct_models)} ({correct_models}) models will be executed"
    )
    if selected_action == 3 and len(correct_models) > 1:
        print(f"The model {correct_models[0]} was selected from your list")
    print("".ljust(width_of_text, "-"))
    need_test = input("The model need to be run and tested in training dataset (y/n)?")
    if need_test[0] == "y":
        return selected_action, tuple(correct_models), True
    return selected_action, tuple(correct_models), False


def check_models(text: str, action: int) -> tuple:
    """_summary_

    Args:
        text (_type_): str to check if all models are correct

    Returns:
        tuple: (incorrect name of  model(s), correct name of model(s) )
    """
    models_splitted = text.upper().split(",")
    models_str = list()
    for item in models_splitted:
        if "-" in item:
            splitted = item.split("-")
            if splitted[1].isdigit():
                models_str.extend(tuple(repeat(splitted[0], int(splitted[1]))))
            else:
                models_str.append(splitted[0])
        else:
            models_str.append(item)
    if action == 1:  # find best parameters
        max_quantity_models = 8
    elif action >= 2:  # Train and test models
        max_quantity_models = 8
    h = tuple(zip(range(1, max_quantity_models + 1), repeat("A", max_quantity_models)))
    k = tuple(zip(range(1, max_quantity_models + 1), repeat("B", max_quantity_models)))
    sc1 = ["".join((str(item[0]), item[1])) for item in h]
    sc2 = ["".join((str(item[0]), item[1])) for item in k]
    sc1.extend(sc2)
    sc1 = sorted(sc1)
    all_combinations = tuple(sc1)
    no_ok = list()
    if "all" in text:
        models_ok = all_combinations
    else:
        models_ok = list()
        no_ok = list()
        for i, item in enumerate(models_str):
            if item.strip() in all_combinations:
                if action == 1:
                    models_ok.append(item.replace("4", "8"))  # rf is model number 8
                else:
                    models_ok.append(item)
            else:
                no_ok.append(i)
    return no_ok, models_ok


def print_sms(*kargs) -> None:
    """_summary_
    Format the message and print it
    Args:
        text (str): text to format the message
    """
    full_text = ""
    for item in kargs:
        full_text += f"{item}"
    full_text.upper().ljust(60, "-")
    result_to_txt.append(full_text)
    print(full_text)


def form_row(values: tuple) -> str:
    """_summary_
    Form a row to be printed
    Args:
        # model_name (str): Name of model
        # mod_range (str): range as str(e.g. 1-100)
        # best_lab (str): best estimator value as str for labeled data (e.g. '14')
        # best_orig(str): best estimator value as str for original data (e.g. '18')
        # max_leaf_lab(str): max number of leaf nodes for labeled data (e.g. '6')
        # max_leaf_orig(str): max number of leaf nodes for original data (e.g. '6')
    Returns:
        str: _description_
    """
    cs = 10
    (
        model_name,
        mod_range_lab,
        best_lab,
        max_leaf_lab,
        mod_range_ori,
        best_orig,
        max_leaf_orig,
    ) = values
    x = f"""|{model_name.center(cs, ' ')}|{mod_range_lab.center(cs, ' ')}|{best_lab.center(cs, ' ')}|
    {max_leaf_lab.center(cs, ' ')}|{mod_range_ori.center(cs, ' ')}|{best_orig.center(cs, ' ')}
    |{max_leaf_orig.center(cs, ' ')}|"""
    return x


def print_estimators_table():
    """
    Print estimator table
    """
    est_table = get_estimators_table()[-1]
    print("".center(width_of_text - 3, "-"))
    cols_name = ("Model", "RgLB", "BELD", "MLLD", "RgOR", "BEOD", "MLOD")
    print(form_row(cols_name))
    est_table = get_estimators_table()[-1]
    tmp_list = list()
    print("".center(width_of_text - 3, "-"))
    for item in est_table.index:
        tmp_list = list(est_table.loc[item, :])
        tmp_list.insert(0, item)
        all_values = tmp_list
        all_values_str = tuple([str(val) for val in all_values])
        txt_print = form_row(all_values_str)
        print(txt_print)
    print("".center(width_of_text - 3, "-"))
    print("RgLB --> Range for model(s) with labeled data")
    print("BELD --> Best estimator with labeled data")
    print("MLLD --> Max number of leaf nodes labeled data")
    print("RgLB --> Range for model(s) with original data")
    print("BEOD --> Best estimator with original data ")
    print("MLOD --> Max number of leaf nodes original data")


def save_model(model, file_path):
    """
    Save ML model into a json file
    Args:
        Model (ML object): Instanace of ML object
        file_path (str): Path of the file to be saved
    Returns: None
    """
    _saved_model = {}
    _saved_model["C"] = (model.C,)
    _saved_model["max_iter"] = (model.max_iter,)
    _saved_model["solver"] = (model.solver,)
    _saved_model["X_train"] = (
        model.X_train.tolist() if model.X_train is not None else "None",
    )
    _saved_model["y_train"] = (
        model.y_train.tolist() if model.y_train is not None else "None"
    )
    json_txt = json.dumps(_saved_model, indent=4)
    with open(file_path, "w") as file:
        file.write(json_txt)


def load_json_model(file_path):
    """
    Load ML model from file

    Args:
    file_path(str): File pathof the file with model saved

    Return:
        An instance of ML model from file
    """
    with open(file_path, "r") as model:
        return model


def train_and_test(models_to_run: tuple, file_number: 0) -> None:
    """_summary_
    Train model(s)
    Args:
        models_to_run (tuple): name of models to run
    """
    all_acc_tmp = list()
    pos_acc_tmp = list()
    all_mod = list()
    for mod in models_to_run:
        model = models.get(int(mod[0]))
        if mod[1] == "A":
            j = 1
        else:
            j = 2
        data_to_use = k_data.get(j)
        print_sms(f"Running model '{model}' with '{data_to_use}' dataset")
        dataset = process_data(data_to_use)
        print(
            "The best estimator for each model was computed by GridSearchCV\n"
            "and are shown in following table:"
        )
        print_estimators_table()
        trained_model = train_model(model, full_data=dataset, k_data=data_to_use)
        if not os.path.exists("models_saved"):
            os.mkdir("models_saved")
        print(f"Saving model {mod}-{model}")
        # try:
        # save_model(train_model,  f"models_saved/run-{file_number}-{mod}-{model}.json")
        with open(f"models_saved/run-{file_number}-{mod}-{model}.pkl", "wb") as fp:
            # pickle.dump(trained_model, fp)
            joblib.dump(trained_model, fp)
        # except:
        # pass
        print_sms(
            f"the trained model have been saved as {mod}-{model}.pkl in the folder 'models_saved'"
        )
        results, all_prediction = test_model(trained_model, dataset)
        print_sms("Report of model's performance %s" % mod)
        all_mod.append(model)
        performance = get_model_performance(
            all_prediction_2022=all_prediction, full_data=dataset, all_results=results
        )
        all_res, report = performance
        overall_acc_model = np.average(report.percent_pron_corr)
        all_acc_tmp.append(overall_acc_model)
        positive_acc_model = all_res.mean()["precision"]
        pos_acc_tmp.append(positive_acc_model)
    run_array = [file_number for _ in range(len(all_acc_tmp))]
    all_acc = pd.DataFrame(
        {
            "run": run_array,
            "model": all_mod,
            "ov_acc": overall_acc_model,
            "pos_acc": positive_acc_model,
        }
    )
    result_to_txt.extend([str(item) for item in performance])
    print("".ljust(width_of_text, "-"))
    with open("results.txt", "w") as g:
        g.write("\n".join(result_to_txt))
    return all_acc


def find_best_estimator(models_to_run: tuple, ask_range: False, file_number: 0) -> None:
    """_summary_
    Find best estimators in models to run
    Args:
        models_to_run (tuple): tuple with all models' name to run
        ask_range (bool): True if user have to enter the range
        final_estimator (int): number of
    """
    results_find = list()
    selected_range = ""
    from collections import Counter

    print(models)
    models_of_interest = tuple(
        Counter([models.get(int(item[0])) for item in models_to_run]).keys()
    )
    for md in models_to_run:
        mod = md
        print("Running model ", md)
        if md[1] == "A":
            j = 1
        else:
            j = 2
        if md[0] == "4":
            mod = "8" + mod[1]
        model = models.get(int(mod[0]))
        data_to_use = k_data.get(j)  # relabeled or original
        dataset = process_data(data_to_use)
        # dataset = dataset.iloc[:100,:]
        X_data = dataset.loc[:, "1_week_ago":"8_week_ago"]
        estimators_table = get_estimators_table()[-1]
        if j == 1:
            y_data = dataset["status_clf"]
        else:
            y_data = dataset["status"]
        X, X_test, y, y_test = train_test_split(X_data, y_data)
        print(len(X_data))
        print(
            f"Running model '{model}' with '{data_to_use}' dataset to find best estimator"
        )
        results_find.append(
            f"Running model '{model}' with '{data_to_use}' dataset to find best estimator"
        )
        while selected_range == "":
            if j == 1:
                selected_range = estimators_table.loc[model, "range_labeled"]
            else:
                selected_range = estimators_table.loc[model, "range_original"]
            print("The value of the parameters are:")
            print_estimators_table()
            print(
                "Please enter a range (e.g. 1-20) or leave it in blank to take range of model"
            )
            print(
                f"'{model}' from table. The range will be considered for all models of your"
            )
            if ask_range:
                str_rng = input(f"interest ({' '.join(models_of_interest)}):")
            else:
                str_rng = ""
            if not str_rng.isdigit():
                print(f"The selected range of {model} is:", selected_range)
            else:
                selected_range = f"1-{str_rng}"
        results_find.append(f"The selected range of {model} is:" + selected_range)
        model_range_str = selected_range.split("-")
        model_range = tuple([int(item) for item in model_range_str])
        low_rg, up_rg = model_range
        param_grid = [{"n_estimators": tuple(range(low_rg, up_rg + 1))}]
        # -----
        print("The selected model is", model, "with ", data_to_use, " data")
        selected_model = select_model_to_run(model_name=model, k_data=data_to_use)
        # ----
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        grid_SearCV = GridSearchCV(
            selected_model, param_grid=param_grid, n_jobs=-1, verbose=1, cv=5
        )
        grid_SearCV.fit(X_train, y_train)
        y_predict = cross_val_predict(grid_SearCV, X_train, y_train, cv=5)
        best_est = grid_SearCV.get_params().get("estimator__n_estimators")
        max_leaf = grid_SearCV.best_estimator_.max_features
        cf = confusion_matrix(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        results_find.append("best score: " + str(grid_SearCV.best_score_))
        results_find.append(str(grid_SearCV.get_params()))
        results_find.append(str(cf))
        results_find.append("recall:" + str(recall))
        joblib.dump(grid_SearCV, f"grid_{md}.pkl")
        print("best score:", grid_SearCV.best_score_)
        print("confusion matrix")
        print(cf)
        print(grid_SearCV.get_params())
        print("recall:", recall)
        print("Best number of estimators:", best_est)
        results_find.append("Best number of estimators:" + str(best_est))
        if j == 1:
            estimators_table.at[model, "range_labeled"] = selected_range
            estimators_table.at[model, "best_data_labeled"] = best_est
            estimators_table.at[model, "max_leaf_data_labeled"] = max_leaf
        else:
            estimators_table.at[model, "range_original"] = selected_range
            estimators_table.at[model, "best_data_original"] = best_est
            estimators_table.at[model, "max_leaf_data_original"] = max_leaf
        estimators_table = estimators_table.astype(int, errors="ignore")
        estimators_table.to_csv("best_estimators.csv")
        results_find.append("Final table")
        results_find.append(str(estimators_table))
        print("Max number of leaf nodes:", max_leaf)
        print("The Parameters have been saved")
        print("".ljust(width_of_text, "-"))
        with open(f"{md}-find_best_est.txt", "w") as f:
            f.write("\n".join(results_find))


def get_prediction(selected_model_str: str):
    """_summary_
    Get prediction
    Args:
        selected_model_str (str): name of selected model
    """
    models_sv = os.listdir("models_saved")
    model_file = ""
    for model in models_sv:
        if model[:2] == selected_model_str[0]:
            model_file = model
    print_sms("The model is being loaded")
    model_instance = joblib.load(f"./models_saved/{model_file}")
    print_sms("The instance of selected model was loaded")
    predict_data = pd.read_csv("predict_data.csv")
    data = predict_data.loc[:, "1_week_ago":"8_week_ago"]
    print_sms("Getting prediction ...")
    prediction = model_instance.predict(data)
    res = pd.DataFrame({"Zone": predict_data.MZ, "Prediction": prediction})
    res = res.astype({"Zone": str, "Prediction": int})
    pos_zones = tuple(res[res.Prediction == 1].Zone)
    print(
        f"The following zones (total={len(pos_zones)}) will have TSD in this month\n",
        pos_zones,
    )
    print("".ljust(width_of_text, "-"))


def run_in_HPC(models_to_run: tuple, find_grid: bool, pre_train_test=False):
    """
    Script to run all code in HPC
    models_to_run (tuple): names of models to be runned
    find_grid (bool): True if the model will run any grid algorithm
    pre_train_test(bool): True if the model need to be run and tested in the training set
    """
    with open("model_to_run.txt", "r") as f:
        data = f.read().split(",")[2]
        test_or_not = bool(data)
    for model in models_to_run:
        if not find_grid:
            if test_or_not:
                train_and_test(models_to_run, 0)
            if not pre_train_test:
                all_acc = pd.DataFrame(
                    {"run": [""], "model": [""], "ov_acc": [""], "pos_acc": [""]}
                )
                for i in range(1, 2):
                    models_acc = train_and_test(models_to_run, i)
                    all_acc = pd.concat((all_acc, models_acc))
                    all_acc.to_csv("all_acc.csv", index=True)
                    filename = f"run-{i}-results-{model}.txt"
                    print("Run number ", i, "has been finished")
            else:
                for i in range(1, 2):
                    model = model
                    if model[1] == "A":
                        j = 1
                    else:
                        j = 2
                    data_to_use = k_data.get(j)
                    dataset = process_data(k_data=data_to_use)
                    print(
                        f"Training and testing model in training datasets. Run number {i}"
                    )
                    model_trained = train_model(
                        model,
                        full_data=dataset,
                        k_data=data_to_use,
                        train_test=test_or_not,
                    )
                    joblib.dump(
                        model_trained,
                        f"./models_saved/train_test_model_{model}_run_{i}.pkl",
                    )
                    print(
                        f"The model {model} has been saved, filename: (train_test_model_{model}_run_{i}.pkl)"
                    )
        else:
            find_best_estimator(models_to_run, ask_range=False, file_number=0)
            filename = f"f_best_est-{model}.txt"
            with open(filename, "w") as f:
                f.write("\n".join(result_to_txt))
            print_sms(f"results was saved in in the file '{filename}'")


if __name__ == "__main__":
    print_sms("Welcome to this toolbox")
    # Disable from here to run in HPC
    while True:
        selected_action, models_to_run, need_test = run_tool_box()
        if selected_action == 1:
            print_sms("RUNNING GRIDSEARCH TO FIND BEST PARAMETERS")
            find_best_estimator(models_to_run, ask_range=False, file_number=0)
        elif selected_action == 3:
            get_prediction(models_to_run)
        else:
            if need_test:
                run_in_HPC(models_to_run, False, True)
            else:
                print_sms("TRAINING AND TESTING MODELS")
                train_and_test(models_to_run, 0)

    # Enable from here to run in HPC
    # with open('model_to_run.txt', 'r') as f:
    # models_ro_run, str_action, pre_train_test = f.read().split(',')
    # grid_search = False
    # if str_action == 'find':
    # grid_search = True
    # run_in_HPC((models_ro_run, ), grid_search, pre_train_test)
