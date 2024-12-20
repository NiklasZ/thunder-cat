import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from optuna import create_study, trial
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from configuration import (
    MODEL_FOLDER_PATH,
    cls_str,
    configure_logger,
    current_timestamp,
    filter_junk,
)
from imagenet_classes import MAX_CLASS_RANK

binary_cat_model: lgb.LGBMClassifier | None = None
model_parameters: dict | None = None
logger = configure_logger()


def load_binary_classifier_config(config_path: str, imagenet_model_name: str) -> dict:
    with open(config_path, "r") as file:
        classifier_config = json.load(file)
    if classifier_config["imagenet_classifier_name"] != imagenet_model_name:
        raise (
            f"Binary classifier not configured for supplied imagenet model {imagenet_model_name}. "
            + f'Only supports {classifier_config["imagenet_classifier_name"]}'
        )
    return classifier_config


def load_binary_classifier(model_path: str):
    global binary_cat_model
    if not binary_cat_model:
        with open(model_path, "rb") as file:
            binary_cat_model = pickle.load(file)


def binary_classify_cat_vs_other(class_score_batch: list[np.ndarray]) -> list[float]:
    """Evaluation function using the scores from imagenet and binary classifier to
    determine if it is a cat or not.

    Args:
        class_scores (np.ndarray): classification input

    Returns:
        list[float]: model confidence (probability) that frame is cat
    """
    global binary_cat_model

    prediction = binary_cat_model.predict_proba(np.stack(class_score_batch))
    return [p[1] for p in prediction]


def prepare_data(cat_df: pd.DataFrame, other_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses the dataframes to become usable training data

    Args:
        cat_df (pd.DataFrame): all frames capturing cats
        other_df (pd.DataFrame): all frames capturing not cats

    Returns:
        tuple[np.ndarray, np.ndarray]: features, labels
    """
    cat_df["binary_label"] = 1
    other_df["binary_label"] = 0
    combined_df = pd.concat((cat_df, other_df))
    combined_df = filter_junk(combined_df)
    X = combined_df[[cls_str(idx) for idx in range(MAX_CLASS_RANK)]]
    y = combined_df["binary_label"]
    return X.to_numpy(), y.to_numpy()


def train_binary_classifier(
    cat_df: pd.DataFrame, other_df: pd.DataFrame, false_pos_weight: float, false_neg_weight: float
) -> tuple[str, lgb.LGBMClassifier, np.ndarray, np.ndarray]:
    """Trains a binary classifier, given selected frames of cats and not cats as well as the imagenet
    model's class scores. The training scheme uses a light-gradient boosting machine (lgbm) and runs
    a hyper-parameter tuner (optuna) to find suitable parameters. The optimisation goal is to minimise
    the number of false positives and false negatives, based on the weightings provided.

    Args:
        cat_df (pd.DataFrame): cat data
        other_df (pd.DataFrame): not-cat data
        false_pos_weight (float): penalty weighting for false positives (misidentifying cat)
        false_neg_weight (float): penalty weighting for false negatives (misidentifying not-cat)

    Returns:
        _type_: tuple[str, lgb.LGBMClassifier, np.ndarray, np.ndarray], model name, model, test features, test labels
    """
    imagenet_classifier_name = cat_df["model_name"].iloc[0]
    X, y = prepare_data(cat_df, other_df)
    X_test = y_test = None

    def calculate_loss(confusion_matrix: np.ndarray) -> float:
        false_pos_penalty = false_pos_weight * confusion_matrix[0, 1]
        false_neg_penalty = false_neg_weight * confusion_matrix[1, 0]
        return false_pos_penalty + false_neg_penalty

    def train(
        n_estimators: int,
        learning_rate: float,
        num_leaves: int,
        max_depth: int,
        classification_threshold: float,
    ) -> tuple[np.ndarray, lgb.LGBMClassifier]:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Metrics to aggregate
        conf_matrices = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Model
        model = lgb.LGBMClassifier(
            objective="binary",
            is_unbalance=True,
            verbose=-1,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
        )

        # K-Fold Cross-Validation
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model
            model.fit(X_train, y_train)

            # Predict with threshold adjustment
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > classification_threshold).astype(int)

            # Metrics for this fold
            conf_matrices.append(confusion_matrix(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        # Aggregate Metrics
        total_conf_matrix = np.sum(conf_matrices, axis=0)
        return total_conf_matrix, model

    def objective(t: trial.Trial) -> float:
        model_args = {
            "n_estimators": t.suggest_int("n_estimators", 20, 400),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": t.suggest_int("num_leaves", 15, 150),
            "max_depth": t.suggest_int("max_depth", 3, 15),
            "classification_threshold": t.suggest_float("classification_threshold", 0.9, 0.999),
        }
        conf_matrix, _ = train(**model_args)
        return calculate_loss(conf_matrix)

    study = create_study()
    study.optimize(objective, 300, show_progress_bar=True, n_jobs=3)
    best_params = study.best_params
    # best_params = {
    #     "n_estimators": 141,
    #     "learning_rate": 0.2608993020980031,
    #     "num_leaves": 109,
    #     "max_depth": 14,
    #     "classification_threshold": 0.9986889841905932,
    # }
    logger.info(f"Best tuning parameters:\n {best_params}")
    conf_matrix, model = train(**best_params)
    logger.info(conf_matrix)

    timestamp = current_timestamp()
    model_name = f"{timestamp}_binary_cat_classifier"
    model_file_path = os.path.join(MODEL_FOLDER_PATH, f"{model_name}.pkl")
    with open(model_file_path, "wb") as file:
        pickle.dump(model, file)

    metrics = {
        "imagenet_classifier_name": imagenet_classifier_name,
        "training_parameters": best_params,
        "confusion_matrix": conf_matrix.tolist(),
    }
    parameters_file_path = os.path.join(MODEL_FOLDER_PATH, f"{timestamp}_parameters.json")
    with open(parameters_file_path, "w") as file:
        json.dump(metrics, file)

    return model_name, model, X_test, y_test
