import os
import numpy as np
import pandas as pd

from utils import (ler_config, calcular_eer, calcular_apcer_bpcer, avaliar_dev_teste,
                   confianca, PASTA_PROJETO)
from treino_espacial import CONJUNTOS

PASSO_CURVA = 0.05


def carregar(config, nome):
    pasta = os.path.join(PASTA_PROJETO, config["resultados_dir"])
    a = pd.read_csv(os.path.join(pasta, "espacial", nome, "scores_videos.csv"), dtype={"video": str})
    b = pd.read_csv(os.path.join(pasta, "temporal", nome, "scores_videos.csv"), dtype={"video": str})
    a = a[["video", "split", "tipo", "rotulo", "score"]].rename(columns={"score": "score_a"})
    b = b[["video", "score_temporal"]].rename(columns={"score_temporal": "score_b"})
    juntos = a.merge(b, on="video")
    return juntos[juntos["split"] == "dev"].copy(), juntos[juntos["split"] == "teste"].copy()


def preparar(dev, teste):
    minimo, maximo = dev["score_b"].min(), dev["score_b"].max()
    for t in [dev, teste]:
        t["score_ab"] = (t["score_a"] + ((t["score_b"] - minimo) / (maximo - minimo)).clip(0, 1)) / 2
    limiares = {c: calcular_eer(dev[c].values, dev["rotulo"].values)[1] for c in ["score_a", "score_b", "score_ab"]}
    for t in [dev, teste]:
        t["conf_a"] = confianca(t["score_a"].values, limiares["score_a"])
        t["dec_a"] = (t["score_a"] >= limiares["score_a"]).astype(int)
        t["dec_b"] = (t["score_b"] >= limiares["score_b"]).astype(int)
        t["dec_ab"] = (t["score_ab"] >= limiares["score_ab"]).astype(int)
    return limiares


def guiada(t, limiar_conf):
    consulta = t["conf_a"].values < limiar_conf
    return np.where(consulta, t["dec_ab"], t["dec_a"]), consulta


def acer(decisao, t):
    return calcular_apcer_bpcer(decisao, t["rotulo"].values, t["tipo"].values, 0.5)


def ponto(decisao, consulta, t):
    m = acer(decisao, t)
    return {"consulta_b": np.mean(consulta), "apcer_max": m["apcer_max"], "bpcer": m["bpcer"], "acer": m["acer"]}


def limiar_por_orcamento(dev, fracao):
    return 0.0 if fracao <= 0 else np.quantile(dev["conf_a"], fracao)


def limiar_menor_erro_dev(dev):
    melhor = None
    for lc in np.concatenate([[0.0], np.unique(dev["conf_a"]), [np.inf]]):
        decisao, consulta = guiada(dev, lc)
        chave = (round(acer(decisao, dev)["acer"], 10), consulta.mean())
        if melhor is None or chave < melhor[0]:
            melhor = (chave, lc)
    return melhor[1]


def correcoes(t):
    c = t[t["consulta"]]
    a_certo, ab_certo = c["dec_a"] == c["rotulo"], c["dec_ab"] == c["rotulo"]
    return {"consultados": len(c), "a_errava": int((~a_certo).sum()), "corrigidos": int((~a_certo & ab_certo).sum()),
            "estragados": int((a_certo & ~ab_certo).sum())}


def avaliar_conjunto(config, protocolo, fold):
    nome = f"p{protocolo}" + (f"_f{fold}" if fold else "")
    print(f"\n========== {nome} ==========")
    pasta_saida = os.path.join(PASTA_PROJETO, config["resultados_dir"], "estrategias", nome)
    os.makedirs(pasta_saida, exist_ok=True)
    dev, teste = carregar(config, nome)
    preparar(dev, teste)

    linhas = []
    for estrategia, coluna in [("só A", "score_a"), ("só B", "score_b"), ("A+B sempre", "score_ab")]:
        r = avaliar_dev_teste(dev[coluna].values, dev["rotulo"].values, teste[coluna].values,
                              teste["rotulo"].values, teste["tipo"].values)
        linhas.append({"estrategia": estrategia, **r, "consulta_b": 0.0 if estrategia == "só A" else 1.0})

    orcamento = config["orcamento_consulta"]
    limiar_conf = limiar_por_orcamento(dev, orcamento)
    for t in [dev, teste]:
        t["guiada"], t["consulta"] = guiada(t, limiar_conf)
    linhas.append({"estrategia": "guiada", **acer(teste["guiada"].values, teste),
                   "consulta_b": teste["consulta"].mean(), "limiar_conf": limiar_conf})
    pd.DataFrame(linhas).to_csv(os.path.join(pasta_saida, "metricas_teste.csv"), index=False)

    limiar_erro = limiar_menor_erro_dev(dev)
    pontos = pd.DataFrame([{"ponto": "só A", **ponto(*guiada(teste, 0.0), teste)},
                           {"ponto": "menor erro no dev", **ponto(*guiada(teste, limiar_erro), teste)},
                           {"ponto": f"orçamento {orcamento:.0%}", **ponto(teste["guiada"], teste["consulta"], teste)},
                           {"ponto": "A+B sempre", **ponto(*guiada(teste, np.inf), teste)}])
    pontos.to_csv(os.path.join(pasta_saida, "pontos.csv"), index=False)
    print(pontos.round(4).to_string(index=False))
    corr = correcoes(teste)
    print(f"limiar de confiança: orçamento {limiar_conf:.2f} | menor erro no dev {limiar_erro:.2f} | {corr}")

    curva = []
    for fracao in np.round(np.arange(0, orcamento + 1e-9, PASSO_CURVA), 2):
        lc = limiar_por_orcamento(dev, fracao)
        linha = {"orcamento": fracao, "limiar_conf": lc}
        for split, t in [("dev", dev), ("teste", teste)]:
            decisao, consulta = guiada(t, lc)
            linha[f"consulta_{split}"], linha[f"acer_{split}"] = consulta.mean(), acer(decisao, t)["acer"]
        curva.append(linha)
    pd.DataFrame(curva).to_csv(os.path.join(pasta_saida, "curva_orcamento.csv"), index=False)
    pd.concat([dev, teste]).to_csv(os.path.join(pasta_saida, "decisoes_videos.csv"), index=False)

    info = {"conjunto": nome, "protocolo": protocolo, "fold": fold}
    return ([{**info, **l} for l in linhas], [{**info, **p} for p in pontos.to_dict("records")],
            {**info, "limiar_conf": limiar_conf, "limiar_menor_erro_dev": limiar_erro, **corr})


if __name__ == "__main__":
    config = ler_config()
    resumo, todos_pontos, todas_correcoes = [], [], []
    for protocolo, fold in CONJUNTOS:
        linhas, pontos, corr = avaliar_conjunto(config, protocolo, fold)
        resumo += linhas
        todos_pontos += pontos
        todas_correcoes.append(corr)
    pasta = os.path.join(PASTA_PROJETO, config["resultados_dir"], "estrategias")
    pd.DataFrame(resumo).to_csv(os.path.join(pasta, "resumo.csv"), index=False)
    pd.DataFrame(todos_pontos).to_csv(os.path.join(pasta, "pontos.csv"), index=False)
    pd.DataFrame(todas_correcoes).to_csv(os.path.join(pasta, "correcoes.csv"), index=False)
    print("\nArquivos salvos em", pasta)
