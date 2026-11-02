from pathlib import Path
import json
import warnings

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

import desenvolvimento_conjunto as d


PASTA = Path(__file__).resolve().parent
SAIDA = PASTA / "resultados_lasso_iqr"

ALPHA = 0.0005
P = 12

VALORES_Q = {
    "nitrate": 3,
    "wind": 4,
    "upwelling": 2,
    "sst": 5,
    "par": 4,
    "mixed": 3,
}


def calcular_metricas(real, previsto):
    return {
        "R2": r2_score(real, previsto),
        "MSE": mean_squared_error(real, previsto),
        "RMSE": root_mean_squared_error(real, previsto),
        "MAE": mean_absolute_error(real, previsto),
    }


def main():
    SAIDA.mkdir(exist_ok=True)

    dados = d.criar_features(d.carregar_dados())

    angulo = (
        2 * np.pi * dados["date"].dt.dayofyear / 365.25
    )
    dados["semiannual_sin"] = np.sin(2 * angulo)
    dados["semiannual_cos"] = np.cos(2 * angulo)

    entradas = [
        f"chlorophyll_lag_{lag}"
        for lag in range(1, P + 1)
    ] + [
        "annual_sin",
        "annual_cos",
        "semiannual_sin",
        "semiannual_cos",
    ]

    for variavel, q in VALORES_Q.items():
        entradas += [
            f"{variavel}_lag_{lag}"
            for lag in range(q + 1)
        ]

    resultados_anuais = []
    todas_previsoes = []
    limites = []

    print("LASSO COM CLIPPING Q3 + 2 IQR", flush=True)
    print("Configuração fixa; apenas 2015–2022.", flush=True)
    print("Alpha:", ALPHA, "| p:", P, flush=True)
    print("q:", VALORES_Q, flush=True)

    for ano in d.ANOS_DESENVOLVIMENTO:
        print(f"A avaliar {ano}...", flush=True)

        treino, teste = d.construir_divisao(
            dados,
            ano_validacao=ano,
        )

        X_treino = treino[entradas].to_numpy(
            dtype=float, copy=True
        )
        X_teste = teste[entradas].to_numpy(
            dtype=float, copy=True
        )

        y_treino = treino[d.ALVO].to_numpy(dtype=float)
        y_teste = teste[d.ALVO].to_numpy(dtype=float)

        if not all(
            np.isfinite(array).all()
            for array in [
                X_treino, X_teste, y_treino, y_teste
            ]
        ):
            raise ValueError("Existem valores não finitos.")

        # Limite comum aos lags de clorofila, aprendido
        # exclusivamente nos alvos das linhas de treino.
        q1, q3 = np.quantile(y_treino, [0.25, 0.75])
        iqr = q3 - q1
        limite = q3 + 2.0 * iqr

        # Contar as entradas afetadas antes do clipping.
        afetadas_treino = X_treino[:, :P] > limite
        afetadas_teste = X_teste[:, :P] > limite

        limites.append({
            "ano_validacao": ano,
            "Q1_treino": q1,
            "Q3_treino": q3,
            "IQR_treino": iqr,
            "limite_superior": limite,
            "percentagem_entradas_lag_limitadas_treino":
                100 * afetadas_treino.mean(),
            "percentagem_entradas_lag_limitadas_validacao":
                100 * afetadas_teste.mean(),
            "dias_validacao_com_algum_lag_limitado":
                int(afetadas_teste.any(axis=1).sum()),
        })

        # Alterar apenas os lags, nunca os alvos.
        X_treino[:, :P] = np.minimum(
            X_treino[:, :P], limite
        )
        X_teste[:, :P] = np.minimum(
            X_teste[:, :P], limite
        )

        # Normalização aprendida depois do clipping,
        # exclusivamente no treino.
        medias = X_treino.mean(axis=0)
        escalas = X_treino.std(axis=0, ddof=0)
        escalas = np.where(escalas == 0, 1.0, escalas)

        X_treino = (X_treino - medias) / escalas
        X_teste = (X_teste - medias) / escalas

        modelo = Lasso(
            alpha=ALPHA,
            max_iter=50000,
            tol=1e-5,
        )

        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            modelo.fit(X_treino, y_treino)

        if any(
            issubclass(a.category, ConvergenceWarning)
            for a in avisos
        ):
            raise RuntimeError(
                f"O LASSO não convergiu no ano {ano}."
            )

        previsto = np.maximum(
            modelo.predict(X_teste), 0.0
        )

        if not np.isfinite(previsto).all():
            raise ValueError("Existem previsões não finitas.")

        resultados_anuais.append({
            "ano": ano,
            "n_treino": len(treino),
            "n_validacao": len(teste),
            **calcular_metricas(y_teste, previsto),
        })

        todas_previsoes.append(pd.DataFrame({
            "date": teste["date"].to_numpy(),
            "ano": ano,
            "real": y_teste,
            "previsto": previsto,
            "residuo": y_teste - previsto,
            "erro_quadrado": (y_teste - previsto) ** 2,
        }))

    anuais = pd.DataFrame(resultados_anuais)
    previsoes = (
        pd.concat(todas_previsoes, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    tabela_limites = pd.DataFrame(limites)

    if previsoes["date"].duplicated().any():
        raise ValueError("Existem datas de validação repetidas.")

    globais = pd.DataFrame([{
        "modelo": "LASSO_IQR",
        "n": len(previsoes),
        **calcular_metricas(
            previsoes["real"].to_numpy(),
            previsoes["previsto"].to_numpy(),
        ),
    }])

    mse_ponderado = np.average(
        anuais["MSE"],
        weights=anuais["n_validacao"],
    )

    np.testing.assert_allclose(
        globais.iloc[0]["MSE"],
        mse_ponderado,
    )

    resumo_anual = anuais[
        ["R2", "MSE", "RMSE", "MAE"]
    ].agg(["mean", "std"])

    configuracao = {
        "metodo": "LASSO",
        "alpha": ALPHA,
        "p": P,
        "q": VALORES_Q,
        "harmonicos": 2,
        "clipping": "Q3 + 2 * (Q3 - Q1)",
        "clipping_aplicado": "lags de clorofila",
        "alvo_alterado": False,
        "entradas": entradas,
    }

    anuais.to_csv(
        SAIDA / "metricas_por_ano.csv", index=False
    )
    globais.to_csv(
        SAIDA / "metricas_globais.csv", index=False
    )
    resumo_anual.to_csv(
        SAIDA / "resumo_metricas_anuais.csv"
    )
    previsoes.to_csv(
        SAIDA / "previsoes_fora_treino.csv", index=False
    )
    tabela_limites.to_csv(
        SAIDA / "limites_por_divisao.csv", index=False
    )

    with (SAIDA / "configuracao.json").open(
        "w", encoding="utf-8"
    ) as ficheiro:
        json.dump(
            configuracao,
            ficheiro,
            indent=2,
            ensure_ascii=False,
        )

    print("\nMÉTRICAS GLOBAIS")
    print(
        globais.to_string(
            index=False, float_format="%.6f"
        )
    )

    print("\nMÉTRICAS POR ANO")
    print(
        anuais.to_string(
            index=False, float_format="%.6f"
        )
    )

    print("\nMÉDIA E DESVIO-PADRÃO DAS MÉTRICAS ANUAIS")
    print(resumo_anual.to_string(float_format="%.6f"))

    print("\nLIMITES POR DIVISÃO")
    print(
        tabela_limites.to_string(
            index=False, float_format="%.4f"
        )
    )

    print("\nCONFIRMADO: MSE global = MSE anual ponderado.")
    print("Avaliação de desenvolvimento de configuração fixa.")
    print("Nenhum modelo de submissão foi criado.")
    print("\nResultados guardados em:", SAIDA)


if __name__ == "__main__":
    main()