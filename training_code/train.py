import argparse
from pathlib import Path
import pickle
import sys
import warnings
import subprocess

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)


SUBMISSION = Path(__file__).resolve().parents[1]
BEST_MODEL = SUBMISSION / "best_model"

sys.path.insert(0, str(BEST_MODEL))

from utils import (
    TARGET,
    P,
    ENVIRONMENTAL,
    build_training_features,
    build_prediction_features,
    calculate_iqr_limit,
    feature_names,
    transform_features,
)


ALPHA = 0.0005
INITIAL_ROWS = 18


def fit_model(X, y):
    q1, q3, limit = calculate_iqr_limit(y)

    clipped = np.array(X, dtype=float, copy=True)
    clipped[:, :P] = np.minimum(
        clipped[:, :P], limit
    )

    means = clipped.mean(axis=0)
    scales = clipped.std(axis=0, ddof=0)
    scales = np.where(scales == 0, 1.0, scales)

    normalized = transform_features(
        X, limit, means, scales
    )

    estimator = Lasso(
        alpha=ALPHA,
        max_iter=50000,
        tol=1e-5,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(normalized, y)

    if any(
        issubclass(item.category, ConvergenceWarning)
        for item in caught
    ):
        raise RuntimeError("O LASSO não convergiu.")

    return {
        "estimator": estimator,
        "feature_names": feature_names(),
        "chlorophyll_limit": limit,
        "means": means,
        "scales": scales,
        "configuration": {
            "method": "LASSO",
            "alpha": ALPHA,
            "p": P,
            "q": {
                short_name: q
                for short_name, _, q in ENVIRONMENTAL
            },
            "harmonics": 2,
            "clipping": "Q3 + 2 * (Q3 - Q1)",
            "initial_rows_discarded": INITIAL_ROWS,
        },
        "training_quartiles": {
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
        },
    }


def predict_matrix(model, X):
    transformed = transform_features(
        X,
        model["chlorophyll_limit"],
        model["means"],
        model["scales"],
    )

    predicted = np.maximum(
        model["estimator"].predict(transformed),
        0.0,
    )

    if not np.isfinite(predicted).all():
        raise ValueError("Existem previsões não finitas.")

    return predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=Path,
        help="Caminho do dataset original XLSX ou CSV.",
    )
    parser.add_argument(
    "--reproduce-development",
     action="store_true",
     help="Reproduzir as avaliações da configuração final.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help=(
            "Executar um script histórico de development pelo nome. "
            "Pode ser repetido para executar vários scripts."
        ),
    )

    args = parser.parse_args()

    args.data_path = args.data_path.resolve()

    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)

    development_dir = Path(__file__).resolve().parent / "development"

    experiments = list(args.experiment)

    if args.reproduce_development:
        experiments += [
            "avaliar_lasso_iqr.py",
            "avaliar_lasso_iqr_2023.py",
        ]

    # Aceitar apenas scripts existentes na pasta entregue.
    available = {
        path.name: path
        for path in development_dir.glob("*.py")
    }

    for name in experiments:
        if name not in available:
            raise ValueError(
                f"Experiência desconhecida: {name}. "
                f"Disponíveis: {', '.join(sorted(available))}"
            )

    # Preservar a ordem e evitar execuções duplicadas.
    for name in dict.fromkeys(experiments):
        print(
            f"\nA reproduzir desenvolvimento: {name}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(available[name]),
                str(args.data_path),
            ],
            cwd=str(development_dir),
            check=True,
        )

    print(
        "\nConfiguração final fixada pelo grupo: "
        "LASSO alpha=0.0005, p=12, dois harmónicos "
        "e clipping Q3 + 2 IQR.",
        flush=True,
    )


    if args.data_path.suffix.lower() == ".csv":
        raw = pd.read_csv(args.data_path)
    else:
        raw = pd.read_excel(
            args.data_path,
            sheet_name="daily_data",
        )

    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.loc[
        raw["date"].dt.year.between(2015, 2023)
    ].copy()

    daily, features = build_training_features(raw)

    # Preservar a mesma regra usada nas avaliações anteriores.
    eligible = np.arange(len(daily)) >= INITIAL_ROWS

    X = features.to_numpy(dtype=float)
    y = daily[TARGET].to_numpy(dtype=float)

    if not (
        np.isfinite(X[eligible]).all()
        and np.isfinite(y[eligible]).all()
    ):
        raise ValueError("Existem valores não finitos nos dados.")

    development = eligible & (
        daily["date"].dt.year <= 2022
    ).to_numpy()

    evaluation = eligible & (
        daily["date"].dt.year == 2023
    ).to_numpy()

    if development.sum() == 0 or evaluation.sum() != 365:
        raise ValueError("Divisão de validação inesperada.")

    # Confirmar que cada vetor construído para previsão
    # coincide com a linha da matriz de treino.
    current_columns = [
        column for column in daily.columns
        if column != TARGET
    ]

    for position in np.flatnonzero(evaluation):
        vector = build_prediction_features(
            daily.iloc[:position],
            daily.iloc[[position]][current_columns],
        )

        np.testing.assert_allclose(
            vector[0],
            X[position],
            rtol=1e-12,
            atol=1e-12,
        )

    print(
        "CONFIRMADO: entradas de treino e previsão coincidem."
    )

    # Verificação local: treino 2015–2022, avaliação 2023.
    local_model = fit_model(
        X[development], y[development]
    )
    predicted = predict_matrix(
        local_model, X[evaluation]
    )

    print("\nVERIFICAÇÃO LOCAL — 2023")
    print(f"R2: {r2_score(y[evaluation], predicted):.6f}")
    print(
        "MSE:",
        f"{mean_squared_error(y[evaluation], predicted):.6f}",
    )
    print(
        "RMSE:",
        f"{root_mean_squared_error(y[evaluation], predicted):.6f}",
    )
    print(
        "MAE:",
        f"{mean_absolute_error(y[evaluation], predicted):.6f}",
    )
    print(
        "Limite IQR:",
        f"{local_model['chlorophyll_limit']:.6f}",
    )

    # Treino final: todos os dados disponibilizados.
    final_model = fit_model(
        X[eligible], y[eligible]
    )

    final_model["training_start"] = str(
        daily.loc[eligible, "date"].min().date()
    )
    final_model["training_end"] = str(
        daily.loc[eligible, "date"].max().date()
    )
    final_model["training_samples"] = int(eligible.sum())

    destination = BEST_MODEL / "model.pkl"
    temporary = destination.with_suffix(".tmp")

    with temporary.open("wb") as handle:
        pickle.dump(
            final_model,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary.replace(destination)

    print("\nTREINO FINAL — 2015–2023")
    print("Amostras:", final_model["training_samples"])
    print("Entradas:", len(final_model["feature_names"]))
    print(
        "Limite Q3 + 2 IQR:",
        f"{final_model['chlorophyll_limit']:.6f}",
    )
    print("Modelo guardado em:", destination)


if __name__ == "__main__":
    main()