import os
import numpy as np
import pandas as pd

from utils import (ler_config, ler_protocolo, info_video, calcular_ear, estados_olho,
                   atributos_janelas, calcular_eer, calcular_auc, avaliar_dev_teste, PASTA_PROJETO)
from treino_espacial import CONJUNTOS

JANELAS = [5, 10]


def fracao_piscada(nome, config):
    landmarks = np.load(os.path.join(config["preproc_dir"], "landmarks", nome + ".npy"))
    _, _, ear = calcular_ear(landmarks)
    if np.isnan(ear).all():
        return None
    estados = estados_olho(ear, config["fator_olho_fechado"])
    fracoes = {}
    for tamanho in JANELAS:
        janelas = pd.DataFrame(atributos_janelas(ear, estados, tamanho, config["max_quadros_piscada"]))
        validas = janelas[janelas["valida"] == 1]
        if len(validas) == 0:
            return None
        fracoes[f"frac_janelas_piscada_{tamanho}"] = validas["piscada"].mean()
    return fracoes


def tabela_split(nomes, split, config, cache):
    linhas = []
    for nome in nomes:
        if nome not in cache:
            cache[nome] = fracao_piscada(nome, config)
        if cache[nome] is not None:
            linhas.append({"video": nome, "split": split, "tipo": info_video(nome)["tipo"],
                           "rotulo": info_video(nome)["rotulo"], **cache[nome]})
    return pd.DataFrame(linhas)


def auc_por_tipo(scores, tabela):
    resultado = {}
    for tipo in ["print", "replay"]:
        filtro = tabela["tipo"].isin(["real", tipo]).values
        resultado[f"auc_real_x_{tipo}"] = calcular_auc(scores[filtro], tabela["rotulo"].values[filtro])
    return resultado


def avaliar_conjunto(config, protocolo, fold, cache):
    nome = f"p{protocolo}" + (f"_f{fold}" if fold else "")
    print(f"\n========== {nome} ==========")
    pasta_saida = os.path.join(PASTA_PROJETO, config["resultados_dir"], "temporal", nome)
    os.makedirs(pasta_saida, exist_ok=True)

    nomes_dev, _ = ler_protocolo(config, protocolo, "Dev", fold)
    nomes_teste, _ = ler_protocolo(config, protocolo, "Test", fold)
    dev = tabela_split(nomes_dev, "dev", config, cache)
    teste = tabela_split(nomes_teste, "teste", config, cache)
    print(f"Vídeos sem janelas válidas: dev {len(nomes_dev) - len(dev)}, teste {len(nomes_teste) - len(teste)}")

    candidatas = []
    for tamanho in JANELAS:
        eer, _ = calcular_eer(dev[f"frac_janelas_piscada_{tamanho}"].values, dev["rotulo"].values)
        candidatas.append((eer, tamanho))
        print(f"  janela {tamanho:2d}: EER dev {eer:.2%}")
    _, tamanho = min(candidatas)
    print(f"-> janela escolhida no dev: {tamanho}")

    dev["score_temporal"] = dev[f"frac_janelas_piscada_{tamanho}"]
    teste["score_temporal"] = teste[f"frac_janelas_piscada_{tamanho}"]
    r = avaliar_dev_teste(dev["score_temporal"].values, dev["rotulo"].values,
                          teste["score_temporal"].values, teste["rotulo"].values, teste["tipo"].values)
    r.update(auc_por_tipo(teste["score_temporal"].values, teste))
    print(" | ".join(f"{chave} {valor:.4f}" for chave, valor in r.items()))

    pd.DataFrame([r]).to_csv(os.path.join(pasta_saida, "metricas_teste.csv"), index=False)
    pd.concat([dev, teste]).to_csv(os.path.join(pasta_saida, "scores_videos.csv"), index=False)
    return {"conjunto": nome, "protocolo": protocolo, "fold": fold, "janela": tamanho, **r}


if __name__ == "__main__":
    config = ler_config()
    cache = {}
    resumo = pd.DataFrame([avaliar_conjunto(config, protocolo, fold, cache)
                           for protocolo, fold in CONJUNTOS])
    resumo.to_csv(os.path.join(PASTA_PROJETO, config["resultados_dir"], "temporal", "resumo.csv"), index=False)

    colunas = ["janela", "apcer_print", "apcer_replay", "bpcer", "acer", "eer_teste",
               "auc_real_x_print", "auc_real_x_replay"]
    print("\nResumo do ramo B sozinho no teste (limiar do EER do dev):")
    print(resumo[["conjunto"] + colunas].round(4).to_string(index=False))

    print("\nMédia ± desvio por protocolo:")
    for protocolo, grupo in resumo.groupby("protocolo"):
        texto = " | ".join(f"{c} {grupo[c].mean():.3f} ± {grupo[c].std(ddof=0):.3f}"
                           for c in ["acer", "eer_teste", "auc_real_x_print", "auc_real_x_replay"])
        print(f"  P{protocolo} ({len(grupo)}): {texto}")
