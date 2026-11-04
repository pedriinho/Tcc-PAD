import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import (ler_config, ler_protocolo, info_video, criar_modelo,
                   avaliar_dev_teste, confianca, PASTA_PROJETO)
from treino_espacial import DatasetRecortes, CONJUNTOS, EPOCAS, ja_treinado


def pontuar_quadros(modelo, nomes, pasta_recortes, split, dispositivo, passo):
    dados = DatasetRecortes(nomes, pasta_recortes, passo)
    carregador = DataLoader(dados, 128, shuffle=False, num_workers=4)
    scores = []
    with torch.no_grad():
        for x, _, _ in carregador:
            scores.extend(torch.sigmoid(modelo(x.to(dispositivo)).squeeze(1)).cpu().numpy())
    linhas = []
    for (caminho, _, i), s in zip(dados.itens, scores):
        quadro = int(os.path.basename(caminho)[:-4])
        linhas.append({"video": nomes[i], "split": split, "quadro": quadro, "score": float(s)})
    return pd.DataFrame(linhas)


def score_video(df):
    return df.groupby("video")["score"].mean()


def listas(scores, nomes):
    usados = [v for v in nomes if v in scores.index]
    return ([scores[v] for v in usados], [info_video(v)["rotulo"] for v in usados],
            [info_video(v)["tipo"] for v in usados])


def avaliar_modelo(config, protocolo, fold, dispositivo):
    nome = f"espacial_p{protocolo}" + (f"_f{fold}" if fold else "")
    caminho_modelo = os.path.join(PASTA_PROJETO, "modelos", nome + ".pt")
    caminho_log = os.path.join(PASTA_PROJETO, config["resultados_dir"], "treino", nome + "_log.csv")
    print(f"\n========== {nome} ==========")
    if not ja_treinado(caminho_modelo, caminho_log, EPOCAS):
        print("Modelo não terminou o treino (etapa 6), pulando")
        return None
    pasta_saida = os.path.join(PASTA_PROJETO, config["resultados_dir"], "espacial", nome[len("espacial_"):])
    os.makedirs(pasta_saida, exist_ok=True)

    modelo = criar_modelo(pre_treinado=False)
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=dispositivo))
    modelo.to(dispositivo).eval()

    pasta_recortes = os.path.join(config["preproc_dir"], "recortes")
    passo = config["passo_quadros"]
    nomes_dev, _ = ler_protocolo(config, protocolo, "Dev", fold)
    nomes_teste, _ = ler_protocolo(config, protocolo, "Test", fold)
    df = pd.concat([pontuar_quadros(modelo, nomes_dev, pasta_recortes, "dev", dispositivo, passo),
                    pontuar_quadros(modelo, nomes_teste, pasta_recortes, "teste", dispositivo, passo)])
    df.to_csv(os.path.join(pasta_saida, "scores_quadros.csv"), index=False)
    df_dev, df_teste = df[df["split"] == "dev"], df[df["split"] == "teste"]
    print(f"Vídeos sem recorte: dev {len(nomes_dev) - df_dev['video'].nunique()},"
          f" teste {len(nomes_teste) - df_teste['video'].nunique()} (ficam fora das métricas)")

    sd, rd, _ = listas(score_video(df_dev), nomes_dev)
    st, rt, tt = listas(score_video(df_teste), nomes_teste)
    r = avaliar_dev_teste(sd, rd, st, rt, tt)
    print(" | ".join(f"{chave} {valor:.4f}" for chave, valor in r.items()))

    limiar = r["limiar_dev"]
    por_video = []
    for split, d, nomes in [("dev", df_dev, nomes_dev), ("teste", df_teste, nomes_teste)]:
        scores = score_video(d)
        for v in nomes:
            if v not in scores.index:
                continue
            decisao = 1 if scores[v] >= limiar else 0
            por_video.append({"video": v, "split": split, "tipo": info_video(v)["tipo"],
                              "rotulo": info_video(v)["rotulo"], "score": scores[v],
                              "confianca": confianca(scores[v], limiar), "decisao": decisao,
                              "acertou": int(decisao == info_video(v)["rotulo"])})
    pd.DataFrame(por_video).to_csv(os.path.join(pasta_saida, "scores_videos.csv"), index=False)

    return {"modelo": nome, "protocolo": protocolo, "fold": fold, **r}


if __name__ == "__main__":
    config = ler_config()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    resumo = []
    for protocolo, fold in CONJUNTOS:
        linha = avaliar_modelo(config, protocolo, fold, dispositivo)
        if linha is not None:
            resumo.append(linha)

    resumo = pd.DataFrame(resumo)
    resumo.to_csv(os.path.join(PASTA_PROJETO, config["resultados_dir"], "espacial", "resumo.csv"), index=False)
    colunas = ["apcer_print", "apcer_replay", "apcer_max", "bpcer", "acer", "eer_teste", "auc_teste"]
    print("\nResumo no teste (todos os recortes, limiar do EER do dev):")
    print(resumo[["modelo"] + colunas].round(4).to_string(index=False))

    print("\nMédia ± desvio por protocolo:")
    for protocolo, grupo in resumo.groupby("protocolo"):
        texto = " | ".join(f"{c} {100 * grupo[c].mean():.2f} ± {100 * grupo[c].std(ddof=0):.2f}"
                           for c in ["apcer_max", "bpcer", "acer", "eer_teste"])
        print(f"  P{protocolo} ({len(grupo)} modelo(s)): {texto}")
