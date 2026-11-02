import pickle

import numpy as np

from utils import (
    build_prediction_features,
    feature_names,
    transform_features,
)


def load_model(model_path):
    """Carregar o LASSO e os parâmetros aprendidos no treino."""
    with open(model_path, "rb") as ficheiro:
        model = pickle.load(ficheiro)

    if model["feature_names"] != feature_names():
        raise ValueError(
            "As entradas guardadas não correspondem ao código."
        )

    return model


def predict(model, history_df, day_features_df):
    """Prever a clorofila do dia atual sem voltar a treinar."""
    features = build_prediction_features(
        history_df,
        day_features_df,
    )

    transformed = transform_features(
        features,
        model["chlorophyll_limit"],
        model["means"],
        model["scales"],
    )

    prediction = np.asarray(
        model["estimator"].predict(transformed),
        dtype=float,
    ).reshape(-1)

    if prediction.shape != (1,):
        raise ValueError("A previsão deve ter forma (1,).")

    if not np.isfinite(prediction).all():
        raise ValueError("A previsão não é finita.")

    return np.maximum(prediction, 0.0)