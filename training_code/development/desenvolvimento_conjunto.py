from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
)

PASTA = Path(__file__).resolve().parent
FICHEIRO = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else None
)

ALVO = "chlorophyll_a_mg_m3"
ANOS_DESENVOLVIMENTO = list(range(2015, 2023))


def carregar_dados():
    if FICHEIRO is None:
        raise ValueError(
            "Indica o caminho do dataset XLSX como argumento."
        )

    if not FICHEIRO.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {FICHEIRO}"
        )
    df = pd.read_excel(FICHEIRO, sheet_name="daily_data")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Separar o desenvolvimento antes de construir as features.
    desenvolvimento = df[
        df["date"].dt.year.isin(ANOS_DESENVOLVIMENTO)
    ].copy()

    return desenvolvimento.reset_index(drop=True)


def criar_features(dados):
    df = dados.copy()
    novas_colunas = {}

    # Até 18 dias anteriores de clorofila.
    for lag in range(1, 19):
        novas_colunas[f"chlorophyll_lag_{lag}"] = (
            df[ALVO].shift(lag)
        )

    # Variáveis ambientais consideradas nos dois trabalhos.
    variaveis = {
        "nitrate": "nitrate_umol_l",
        "wind": "wind_speed_m_s",
        "upwelling": "upwelling_index",
        "sst": "sst_c",
        "par": "par_umol_m2_s",
        "mixed": "mixed_layer_depth_m",
        "river": "river_discharge_index",
        "salinity": "salinity_psu",
        "current": "current_speed_m_s",
        "cloud": "cloud_fraction",
        "pressure": "surface_pressure_hpa",
        "turbidity": "turbidity_ntu",
    }

    # Lag 0 ambiental é permitido: o valor desse dia é conhecido.
    for nome, coluna in variaveis.items():
        for lag in range(0, 8):
            novas_colunas[f"{nome}_lag_{lag}"] = (
                df[coluna].shift(lag)
            )

    # Representação contínua do ciclo anual.
    angulo = 2 * np.pi * df["date"].dt.dayofyear / 365.25

    novas_colunas["annual_sin"] = np.sin(angulo)
    novas_colunas["annual_cos"] = np.cos(angulo)

    df = pd.concat(
        [df, pd.DataFrame(novas_colunas, index=df.index)],
        axis=1,
    )

    # Mesmas datas disponíveis para todos os candidatos.
    df = df.iloc[18:].reset_index(drop=True)

    return df

def definir_configuracoes():
    variaveis_h = [
        "nitrate", "wind", "upwelling",
        "sst", "par", "mixed", "river",
    ]

    grupos = {
        "original": None,
        "E": ["nitrate", "wind", "upwelling"],
        "H": variaveis_h.copy(),
        "H_sem_PAR": [
            v for v in variaveis_h
            if v != "par"
        ],
        "H_reduzida": [
            "nitrate", "wind", "sst", "mixed", "river",
        ],
    }

    perfis = {
        "compacto": {
            "nitrate": 3,
            "wind": 3,
            "upwelling": 3,
            "sst": 5,
            "par": 5,
            "mixed": 3,
            "river": 5,
            "salinity": 3,
            "current": 3,
            "cloud": 3,
            "pressure": 3,
            "turbidity": 3,
        },
        "curto": {
            "nitrate": 3,
            "wind": 3,
            "upwelling": 3,
            "sst": 7,
            "par": 7,
            "mixed": 3,
            "river": 7,
            "salinity": 7,
            "current": 7,
            "cloud": 7,
            "pressure": 7,
            "turbidity": 7,
        },
    }

    configuracoes = []

    for grupo, variaveis in grupos.items():
        if grupo == "original":
            nomes_perfis = ["original"]
            valores_p = [1, 3, 7, 10, 15]

        elif grupo == "sem_ambientais":
            # Sem variáveis ambientais, os perfis seriam iguais.
            nomes_perfis = ["sem_exogenas"]
            valores_p = [7, 12, 15, 18]

        else:
            nomes_perfis = list(perfis)
            valores_p = [7, 12, 15, 18]

        for perfil in nomes_perfis:
            for p in valores_p:
                entradas = [
                    f"chlorophyll_lag_{lag}"
                    for lag in range(1, p + 1)
                ]

                entradas += ["annual_sin", "annual_cos"]

                if grupo == "original":
                    entradas += [
                        "nitrate_lag_3",
                        "wind_lag_2",
                    ]

                else:
                    for variavel in variaveis:
                        q = perfis[perfil][variavel]

                        entradas += [
                            f"{variavel}_lag_{lag}"
                            for lag in range(0, q + 1)
                        ]

                configuracoes.append({
                    "nome": f"{grupo}_{perfil}_p{p}",
                    "grupo": grupo,
                    "perfil": perfil,
                    "p": p,
                    "entradas": entradas,
                })

    return configuracoes

def definir_candidatos(configuracoes):
    opcoes_modelo = [
        ("OLS", 0.0),
        ("Ridge", 1.0),
        ("Ridge", 10.0),
        ("LASSO", 0.0005),
        ("LASSO", 0.001),
        ("LASSO", 0.002),
    ]

    candidatos = []

    for configuracao in configuracoes:
        for modelo, alpha in opcoes_modelo:
            candidatos.append({
                "nome": (
                    f"{configuracao['nome']}"
                    f"_{modelo}_{alpha:g}"
                ),
                "modelo": modelo,
                "alpha": alpha,
                "entradas": configuracao["entradas"].copy(),
                "grupo": configuracao["grupo"],
                "perfil": configuracao["perfil"],
                "p": configuracao["p"],
            })

    return candidatos

def proteger_fronteiras(dados, anos_excluidos):
    manter = pd.Series(True, index=dados.index)

    for ano in anos_excluidos:
        fim_ano = pd.Timestamp(
            year=int(ano),
            month=12,
            day=31,
        )

        limite = fim_ano + pd.DateOffset(days=18)

        linhas_a_retirar = (
            (dados["date"] > fim_ano)
            & (dados["date"] <= limite)
        )

        manter &= ~linhas_a_retirar

    return dados.loc[manter].copy()


def construir_divisao(
    dados,
    ano_validacao,
    ano_externo=None,
):
    # Na validação interna, excluir também o teste externo.
    anos_excluidos = [ano_validacao]

    if ano_externo is not None:
        if ano_externo == ano_validacao:
            raise ValueError(
                "O ano externo e o de validação devem ser diferentes."
            )

        anos_excluidos.append(ano_externo)

    treino = dados[
        ~dados["date"].dt.year.isin(anos_excluidos)
    ].copy()

    treino = proteger_fronteiras(
        treino,
        anos_excluidos,
    )

    validacao = dados[
        dados["date"].dt.year == ano_validacao
    ].copy()

    # A validação interna também não pode usar lags
    # provenientes do teste externo.
    if ano_externo is not None:
        validacao = proteger_fronteiras(
            validacao,
            [ano_externo],
        )

    return treino, validacao

def avaliar_candidato(candidato, treino, validacao):
    entradas = candidato["entradas"]

    X_treino = treino[entradas].to_numpy(dtype=float)
    X_validacao = validacao[entradas].to_numpy(dtype=float)

    y_treino = treino[ALVO].to_numpy(dtype=float)
    y_validacao = validacao[ALVO].to_numpy(dtype=float)

    # Aprender a normalização exclusivamente no treino.
    medias = X_treino.mean(axis=0)
    escalas = X_treino.std(axis=0, ddof=0)
    escalas = np.where(escalas == 0, 1.0, escalas)

    X_treino = (X_treino - medias) / escalas
    X_validacao = (X_validacao - medias) / escalas

    nome_modelo = candidato["modelo"]
    alpha = candidato["alpha"]

    if nome_modelo == "OLS":
        modelo = LinearRegression()

    elif nome_modelo == "Ridge":
        modelo = Ridge(alpha=alpha)

    elif nome_modelo == "LASSO":
        modelo = Lasso(
            alpha=alpha,
            max_iter=50000,
            tol=1e-5,
        )

    else:
        raise ValueError(f"Modelo desconhecido: {nome_modelo}")

    modelo.fit(X_treino, y_treino)

    # Apenas impedir concentrações negativas.
    # Ainda não aplicamos clipping superior aos lags.
    previsoes = np.maximum(
        modelo.predict(X_validacao),
        0.0,
    )

    return {
        "R2": r2_score(y_validacao, previsoes),
        "RMSE": root_mean_squared_error(y_validacao, previsoes),
        "MAE": mean_absolute_error(y_validacao, previsoes),
    }

def selecionar_na_validacao_interna(
    dados,
    candidatos,
    ano_externo,
    pasta_resultados,
):
    resultados = []

    anos_internos = [
        ano
        for ano in ANOS_DESENVOLVIMENTO
        if ano != ano_externo
    ]

    for ano_interno in anos_internos:
        treino, validacao = construir_divisao(
            dados,
            ano_validacao=ano_interno,
            ano_externo=ano_externo,
        )

        print(
            f"A avaliar ano interno {ano_interno} "
            f"(externo reservado: {ano_externo})...",
            flush=True,
        )

        for candidato in candidatos:
            metricas = avaliar_candidato(
                candidato,
                treino,
                validacao,
            )

            resultados.append({
                "ano_externo": ano_externo,
                "ano_interno": ano_interno,
                "candidato": candidato["nome"],
                "n_treino": len(treino),
                "n_validacao": len(validacao),
                **metricas,
            })

    detalhe = pd.DataFrame(resultados)

    ranking = (
        detalhe
        .groupby("candidato", as_index=False)[
            ["R2", "RMSE", "MAE"]
        ]
        .mean()
        .sort_values(
            ["R2", "RMSE", "candidato"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    detalhe.to_csv(
        pasta_resultados / f"inner_detalhe_{ano_externo}.csv",
        index=False,
    )

    ranking.to_csv(
        pasta_resultados / f"inner_ranking_{ano_externo}.csv",
        index=False,
    )

    nome_vencedor = ranking.iloc[0]["candidato"]

    vencedor = next(
        candidato
        for candidato in candidatos
        if candidato["nome"] == nome_vencedor
    )

    return vencedor, ranking

def executar_nested_cv(dados, candidatos, pasta_resultados):
    resultados = []

    for ano_externo in ANOS_DESENVOLVIMENTO:
        print(
            f"\n===== TESTE EXTERNO: {ano_externo} =====",
            flush=True,
        )

        ficheiro_ranking = (
            pasta_resultados
            / f"inner_ranking_{ano_externo}.csv"
        )

        # Aproveitar rankings já calculados, incluindo o de 2020.
        # Só reutilizar enquanto dados, candidatos e regras
        # de validação permanecerem iguais.
        if ficheiro_ranking.exists():
            ranking = pd.read_csv(ficheiro_ranking)

            nome_vencedor = ranking.iloc[0]["candidato"]

            vencedor = next(
                candidato
                for candidato in candidatos
                if candidato["nome"] == nome_vencedor
            )

            print("Ranking interno recuperado.", flush=True)

        else:
            vencedor, ranking = selecionar_na_validacao_interna(
                dados,
                candidatos,
                ano_externo,
                pasta_resultados,
            )

        # Ajustar o vencedor nos outros sete anos.
        treino, teste = construir_divisao(
            dados,
            ano_validacao=ano_externo,
        )

        metricas_modelo = avaliar_candidato(
            vencedor,
            treino,
            teste,
        )

        real = teste[ALVO].to_numpy()
        referencia = teste["chlorophyll_lag_1"].to_numpy()

        metricas_referencia = {
            "R2": r2_score(real, referencia),
            "RMSE": root_mean_squared_error(real, referencia),
            "MAE": mean_absolute_error(real, referencia),
        }

        for metodo, nome, metricas in [
            ("Modelo escolhido", vencedor["nome"], metricas_modelo),
            ("Valor de ontem", "chlorophyll_lag_1", metricas_referencia),
        ]:
            resultados.append({
                "ano_externo": ano_externo,
                "metodo": metodo,
                "candidato": nome,
                "n_treino": len(treino),
                "n_teste": len(teste),
                **metricas,
            })

        # Guardar o progresso após cada ano concluído.
        tabela = pd.DataFrame(resultados)

        tabela.to_csv(
            pasta_resultados / "nested_resultados_externos.csv",
            index=False,
        )

        print("Vencedor:", vencedor["nome"], flush=True)
        print(
            f"R² externo: {metricas_modelo['R2']:.4f}",
            flush=True,
        )

    resumo = (
        tabela
        .groupby("metodo")[["R2", "RMSE", "MAE"]]
        .agg(["mean", "std"])
    )

    resumo.to_csv(
        pasta_resultados / "nested_resumo.csv"
    )

    print("\nRESULTADOS DOS OITO ANOS")
    print(tabela.to_string(index=False, float_format="%.4f"))

    print("\nMÉDIA E DESVIO-PADRÃO DAS MÉTRICAS ANUAIS")
    print(resumo.to_string(float_format="%.4f"))

    return tabela

def main():
    dados = carregar_dados()
    dados = criar_features(dados)

    configuracoes = definir_configuracoes()
    candidatos = definir_candidatos(configuracoes)

    for candidato in candidatos:
        faltam = set(candidato["entradas"]) - set(dados.columns)

        if faltam:
            raise ValueError(f"Colunas em falta: {faltam}")

    pasta_resultados = PASTA / "resultados_222_candidatos"
    pasta_resultados.mkdir(exist_ok=True)

    print("NESTED CV — 222 CANDIDATOS — 2015–2022")
    print("Número de candidatos:", len(candidatos))
    print("2023 permanece fora desta análise.")

    executar_nested_cv(
        dados,
        candidatos,
        pasta_resultados,
    )

def selecionar_configuracao_final(
    dados,
    candidatos,
    pasta_resultados,
):
    resultados_por_ano = []

    # Identificar exatamente os candidatos desta experiência.
    assinatura = pd.DataFrame([
        {
            "nome": c["nome"],
            "modelo": c["modelo"],
            "alpha": c["alpha"],
            "entradas": "|".join(c["entradas"]),
        }
        for c in candidatos
    ]).to_csv(index=False)

    ficheiro_assinatura = pasta_resultados / "candidatos_assinatura.csv"

    if ficheiro_assinatura.exists():
        from io import StringIO

        tabela_guardada = pd.read_csv(
            ficheiro_assinatura,
            dtype=str,
            keep_default_na=False,
        )

        tabela_atual = pd.read_csv(
            StringIO(assinatura),
            dtype=str,
            keep_default_na=False,
        )

        if not tabela_guardada.equals(tabela_atual):
            raise ValueError(
                "O conteúdo dos candidatos mudou. "
                "Precisamos de verificar a diferença "
                "antes de reutilizar os resultados."
            )
    else:
        ficheiro_assinatura.write_text(
            assinatura,
            encoding="utf-8",
        )

    nomes_esperados = {c["nome"] for c in candidatos}

    for ano in ANOS_DESENVOLVIMENTO:
        ficheiro_ano = (
            pasta_resultados / f"selecao_validacao_{ano}.csv"
        )

        if ficheiro_ano.exists():
            tabela_ano = pd.read_csv(ficheiro_ano)

            if (
                len(tabela_ano) != len(candidatos)
                or set(tabela_ano["candidato"]) != nomes_esperados
                or not tabela_ano["ano_validacao"].eq(ano).all()
                or not np.isfinite(
                    tabela_ano[["R2", "RMSE", "MAE"]].to_numpy()
                ).all()
            ):
                raise ValueError(
                    f"Resultados incompletos ou incompatíveis: "
                    f"{ficheiro_ano.name}"
                )

            print(
                f"Ano {ano}: resultados recuperados.",
                flush=True,
            )

        else:
            treino, validacao = construir_divisao(
                dados,
                ano_validacao=ano,
            )

            resultados_ano = []

            print(
                f"\nValidação em {ano}: "
                f"{len(candidatos)} candidatos",
                flush=True,
            )

            for numero, candidato in enumerate(candidatos, start=1):
                metricas = avaliar_candidato(
                    candidato,
                    treino,
                    validacao,
                )

                resultados_ano.append({
                    "ano_validacao": ano,
                    "candidato": candidato["nome"],
                    **metricas,
                })

                if numero % 100 == 0 or numero == len(candidatos):
                    print(
                        f"  Concluídos: {numero}/{len(candidatos)}",
                        flush=True,
                    )

            tabela_ano = pd.DataFrame(resultados_ano)

            # Escrever primeiro num temporário para evitar
            # tratar um ficheiro parcialmente escrito como completo.
            temporario = ficheiro_ano.with_suffix(".tmp")
            tabela_ano.to_csv(temporario, index=False)
            temporario.replace(ficheiro_ano)

        resultados_por_ano.append(tabela_ano)

    detalhe = pd.concat(
        resultados_por_ano,
        ignore_index=True,
    )

    ranking = (
        detalhe
        .groupby("candidato", as_index=False)[
            ["R2", "RMSE", "MAE"]
        ]
        .mean()
        .sort_values(
            ["R2", "RMSE", "candidato"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    detalhe.to_csv(
        pasta_resultados / "selecao_conjunta_detalhe.csv",
        index=False,
    )

    ranking.to_csv(
        pasta_resultados / "selecao_conjunta_ranking.csv",
        index=False,
    )

    print("\nDEZ MELHORES CANDIDATOS")
    print(
        ranking.head(10).to_string(
            index=False,
            float_format="%.6f",
        )
    )

    referencias = [
        "original_original_p7_LASSO_0.001",
        "H_compacto_p12_LASSO_0.0005",
    ]

    print("\nMODELOS DE REFERÊNCIA")
    print(
        ranking[
            ranking["candidato"].isin(referencias)
        ].to_string(index=False, float_format="%.6f")
    )
    
    return ranking

if __name__ == "__main__":
    main()