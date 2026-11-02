from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

import desenvolvimento_conjunto as d
from avaliar_lasso_iqr import ALPHA, P, VALORES_Q, calcular_metricas


PASTA = Path(__file__).resolve().parent
SAIDA = PASTA / "resultados_lasso_iqr_2023"


def main():
    SAIDA.mkdir(exist_ok=True)

    dados = pd.read_excel(
        d.FICHEIRO,
        sheet_name="daily_data",
    )
    dados["date"] = pd.to_datetime(dados["date"])
    dados = dados.sort_values("date").reset_index(drop=True)
    dados = dados.loc[
        dados["date"].dt.year.between(2015, 2023)
    ].copy()

    dados = d.criar_features(dados)

    angulo = (
        2 * np.pi * dados["date"].dt.dayofyear / 365.25
    )
    dados["semiannual_sin"] = np.sin(2 * angulo)
    dados["semiannual_cos"] = np.cos(2 * angulo)

    entradas = [
        f"chlorophyll_lag_{lag}"
        for lag in range(1, P + 1)
    ] + [
        "annual_sin", "annual_cos",
        "semiannual_sin", "semiannual_cos",
    ]

    for variavel, q in VALORES_Q.items():
        entradas += [
            f"{variavel}_lag_{lag}"
            for lag in range(q + 1)
        ]

    treino = dados.loc[
        dados["date"].dt.year.isin(d.ANOS_DESENVOLVIMENTO)
    ].copy()
    teste = dados.loc[
        dados["date"].dt.year == 2023
    ].copy()

    if len(teste) != 365:
        raise ValueError("Esperados 365 dias em 2023.")

    X_treino = treino[entradas].to_numpy(dtype=float, copy=True)
    X_teste = teste[entradas].to_numpy(dtype=float, copy=True)
    y_treino = treino[d.ALVO].to_numpy(dtype=float)
    real = teste[d.ALVO].to_numpy(dtype=float)

    if not all(
        np.isfinite(array).all()
        for array in [X_treino, X_teste, y_treino, real]
    ):
        raise ValueError("Existem valores não finitos.")

    # Quartis aprendidos apenas em 2015–2022.
    q1, q3 = np.quantile(y_treino, [0.25, 0.75])
    limite = q3 + 2 * (q3 - q1)

    X_treino[:, :P] = np.minimum(X_treino[:, :P], limite)
    X_teste[:, :P] = np.minimum(X_teste[:, :P], limite)

    medias = X_treino.mean(axis=0)
    escalas = X_treino.std(axis=0, ddof=0)
    escalas = np.where(escalas == 0, 1.0, escalas)

    modelo = Lasso(
        alpha=ALPHA,
        max_iter=50000,
        tol=1e-5,
    )

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        modelo.fit(
            (X_treino - medias) / escalas,
            y_treino,
        )

    if any(
        issubclass(a.category, ConvergenceWarning)
        for a in avisos
    ):
        raise RuntimeError("O LASSO não convergiu.")

    previsto = np.maximum(
        modelo.predict((X_teste - medias) / escalas),
        0.0,
    )

    if not np.isfinite(previsto).all():
        raise ValueError("Existem previsões não finitas.")

    ontem = teste["chlorophyll_lag_1"].to_numpy(dtype=float)

    resultados = pd.DataFrame([
        {
            "modelo": "LASSO_IQR",
            **calcular_metricas(real, previsto),
        },
        {
            "modelo": "Valor de ontem",
            **calcular_metricas(real, ontem),
        },
    ])

    previsoes = pd.DataFrame({
        "date": teste["date"].to_numpy(),
        "real": real,
        "previsto": previsto,
        "ontem": ontem,
        "residuo": real - previsto,
        "erro_absoluto": np.abs(real - previsto),
    })

    resultados.to_csv(
        SAIDA / "metricas.csv", index=False
    )
    previsoes.to_csv(
        SAIDA / "previsoes.csv", index=False
    )
    pd.DataFrame([{
        "Q1_treino": q1,
        "Q3_treino": q3,
        "IQR_treino": q3 - q1,
        "limite_superior": limite,
    }]).to_csv(SAIDA / "limite_treino.csv", index=False)

    print("LASSO + IQR — AVALIAÇÃO LOCAL EM 2023")
    print("Dias de treino:", len(treino))
    print("Dias de avaliação:", len(teste))
    print(f"Limite Q3 + 2 IQR: {limite:.6f}")

    print("\nMÉTRICAS")
    print(
        resultados.to_string(
            index=False, float_format="%.6f"
        )
    )

    print("\nResultados guardados em:", SAIDA)
    print("Nenhum modelo de submissão foi criado.")


if __name__ == "__main__":
    main()