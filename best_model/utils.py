import numpy as np
import pandas as pd


TARGET = "chlorophyll_a_mg_m3"
P = 12

ENVIRONMENTAL = [
    ("nitrate", "nitrate_umol_l", 3),
    ("wind", "wind_speed_m_s", 4),
    ("upwelling", "upwelling_index", 2),
    ("sst", "sst_c", 5),
    ("par", "par_umol_m2_s", 4),
    ("mixed", "mixed_layer_depth_m", 3),
]


def feature_names():
    names = [
        f"chlorophyll_lag_{lag}"
        for lag in range(1, P + 1)
    ]

    names += [
        "annual_sin",
        "annual_cos",
        "semiannual_sin",
        "semiannual_cos",
    ]

    for short_name, _, q in ENVIRONMENTAL:
        names += [
            f"{short_name}_lag_{lag}"
            for lag in range(q + 1)
        ]

    return names


def prepare_daily_data(data):
    """Copiar e ordenar os dados sem alterar o DataFrame recebido."""
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    if result["date"].isna().any():
        raise ValueError("Existem datas inválidas.")

    if result["date"].duplicated().any():
        raise ValueError("Existem datas repetidas.")

    differences = result["date"].diff().dropna()

    if not differences.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Os dados não formam uma sequência diária.")

    return result


def build_training_features(data):
    """Construir entradas brutas, antes do clipping e normalização."""
    daily = prepare_daily_data(data)
    features = pd.DataFrame(index=daily.index)

    for lag in range(1, P + 1):
        features[f"chlorophyll_lag_{lag}"] = (
            daily[TARGET].shift(lag)
        )

    angle = (
        2 * np.pi * daily["date"].dt.dayofyear / 365.25
    )

    features["annual_sin"] = np.sin(angle)
    features["annual_cos"] = np.cos(angle)
    features["semiannual_sin"] = np.sin(2 * angle)
    features["semiannual_cos"] = np.cos(2 * angle)

    for short_name, column, q in ENVIRONMENTAL:
        for lag in range(q + 1):
            features[f"{short_name}_lag_{lag}"] = (
                daily[column].shift(lag)
            )

    return daily, features[feature_names()]


def build_prediction_features(history_df, day_features_df):
    """Construir uma linha sem usar a clorofila do dia atual."""
    if len(day_features_df) != 1:
        raise ValueError("É necessária uma linha do dia atual.")

    history = prepare_daily_data(history_df)
    day = day_features_df.iloc[0]
    date = pd.Timestamp(day["date"])

    if pd.isna(date):
        raise ValueError("Data de previsão inválida.")

    if len(history) < P:
        raise ValueError("Histórico insuficiente.")

    if not (history["date"] < date).all():
        raise ValueError("O histórico contém datas não permitidas.")

    if history["date"].iloc[-1] != (
        date - pd.Timedelta(days=1)
    ):
        raise ValueError("Falta o dia anterior no histórico.")

    values = [
        float(history[TARGET].iloc[-lag])
        for lag in range(1, P + 1)
    ]

    angle = 2 * np.pi * date.dayofyear / 365.25

    values += [
        np.sin(angle),
        np.cos(angle),
        np.sin(2 * angle),
        np.cos(2 * angle),
    ]

    for _, column, q in ENVIRONMENTAL:
        values.append(float(day[column]))

        values += [
            float(history[column].iloc[-lag])
            for lag in range(1, q + 1)
        ]

    features = np.asarray(
        values, dtype=float
    ).reshape(1, -1)

    if features.shape != (1, len(feature_names())):
        raise ValueError("Número inesperado de entradas.")

    if not np.isfinite(features).all():
        raise ValueError("Existem entradas não finitas.")

    return features


def calculate_iqr_limit(training_target):
    """Aprender Q1, Q3 e o limite superior apenas no treino."""
    values = np.asarray(training_target, dtype=float)

    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Alvo de treino vazio ou não finito.")

    q1, q3 = np.quantile(values, [0.25, 0.75])
    limit = q3 + 2.0 * (q3 - q1)

    return float(q1), float(q3), float(limit)


def transform_features(features, chlorophyll_limit, means, scales):
    """Aplicar clipping aos lags e a normalização do treino."""
    transformed = np.array(
        features, dtype=float, copy=True
    )

    transformed[:, :P] = np.minimum(
        transformed[:, :P],
        chlorophyll_limit,
    )

    transformed = (transformed - means) / scales

    if not np.isfinite(transformed).all():
        raise ValueError("A transformação produziu valores não finitos.")

    return transformed