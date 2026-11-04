import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (ler_config, ler_protocolo, info_video, calcular_ear, estados_olho,
                   achar_piscadas, atributos_janelas, PASTA_PROJETO)

PROTOCOLO = 4
FOLD = 1
SPLIT = "Train"
JANELAS = [5, 10]
FPS = 30
VIDEO_REAL = "1_1_01_1"
VIDEO_PRINT = "1_1_01_2"
FATORES = np.round(np.arange(0.50, 0.96, 0.05), 2)
LIMITE_FALSO_PRINT = 0.05


def escolher_fator(pasta_landmarks, config, caminho):
    ears = []
    for arquivo in sorted(os.listdir(pasta_landmarks)):
        info = info_video(arquivo[:-4])
        if info["usuario"] > 20 or info["tipo"] == "replay":
            continue
        _, _, ear = calcular_ear(np.load(os.path.join(pasta_landmarks, arquivo)))
        if not np.isnan(ear).all():
            ears.append((info["tipo"], ear))

    linhas = []
    for fator in FATORES:
        piscou = {"real": [], "print": []}
        for tipo, ear in ears:
            estados = estados_olho(ear, fator)
            piscou[tipo].append(len(achar_piscadas(estados, config["max_quadros_piscada"])) > 0)
        linhas.append({"fator": fator, "reais_com_piscada": np.mean(piscou["real"]),
                       "prints_com_piscada": np.mean(piscou["print"])})
    tabela = pd.DataFrame(linhas)
    tabela["diferenca"] = tabela["reais_com_piscada"] - tabela["prints_com_piscada"]
    tabela.to_csv(caminho, index=False)

    print(f"Escolha do fator de olho fechado (treino: usuários 1-20, reais e prints):")
    print((tabela.set_index("fator") * 100).round(1).to_string())
    escolhido = tabela[tabela["prints_com_piscada"] <= LIMITE_FALSO_PRINT]["fator"].max()
    print(f"-> maior fator com no máximo {LIMITE_FALSO_PRINT:.0%} de piscadas falsas em print: {escolhido}"
          f" (no config.yaml: {config['fator_olho_fechado']})\n")


def processar_video(caminho_npy, config):
    landmarks = np.load(caminho_npy)
    ear_esq, ear_dir, ear = calcular_ear(landmarks)
    estados = estados_olho(ear, config["fator_olho_fechado"])
    return ear_esq, ear_dir, ear, estados


def resumo_video(nome, ear, estados, config):
    info = info_video(nome)
    piscadas = achar_piscadas(estados, config["max_quadros_piscada"])
    linha = {"video": nome, "tipo": info["tipo"], "quadros": len(ear),
             "quadros_sem_face": int(np.isnan(ear).sum()),
             "ear_mediana": np.nanmedian(ear) if not np.isnan(ear).all() else np.nan,
             "piscadas": len(piscadas), "duracoes": piscadas}
    for tamanho in JANELAS:
        janelas = atributos_janelas(ear, estados, tamanho, config["max_quadros_piscada"])
        linha[f"janelas_{tamanho}"] = len(janelas)
        linha[f"janelas_{tamanho}_com_piscada"] = sum(j["piscada"] for j in janelas)
    return linha


def grafico(pasta_landmarks, config, caminho):
    fig, eixos = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    for eixo, video in zip(eixos, [VIDEO_REAL, VIDEO_PRINT]):
        ear_esq, ear_dir, ear, estados = processar_video(
            os.path.join(pasta_landmarks, video + ".npy"), config)
        tempo = np.arange(len(ear)) / FPS
        referencia = np.array([np.nanmedian(ear[:t + 1]) for t in range(len(ear))])
        eixo.plot(tempo, ear_esq, color="#2a78d6", linewidth=1, alpha=0.5, label="olho esquerdo")
        eixo.plot(tempo, ear_dir, color="#1baf7a", linewidth=1, alpha=0.5, label="olho direito")
        eixo.plot(tempo, ear, color="#2a78d6", linewidth=2, label="média")
        eixo.plot(tempo, config["fator_olho_fechado"] * referencia, color="gray",
                  linestyle="--", linewidth=1, label="limiar de olho fechado")
        fechados = estados == 1
        eixo.scatter(tempo[fechados], ear[fechados], color="#eb6834", s=25, zorder=3,
                     label="quadro com olho fechado")
        piscadas = achar_piscadas(estados, config["max_quadros_piscada"])
        eixo.set_title(f"{video} ({info_video(video)['tipo']}) - {len(piscadas)} piscada(s)")
        eixo.set_ylabel("EAR")
        eixo.grid(alpha=0.3)
    eixos[0].legend(fontsize=8, loc="lower right", ncol=3)
    eixos[1].set_xlabel("tempo (s)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    config = ler_config()
    pasta_landmarks = os.path.join(config["preproc_dir"], "landmarks")
    pasta_saida = os.path.join(PASTA_PROJETO, config["resultados_dir"], "verificacoes", "olhos")
    os.makedirs(pasta_saida, exist_ok=True)
    escolher_fator(pasta_landmarks, config, os.path.join(pasta_saida, "escolha_fator.csv"))

    nomes, _ = ler_protocolo(config, PROTOCOLO, SPLIT, FOLD)
    linhas = []
    for nome in nomes:
        ear_esq, ear_dir, ear, estados = processar_video(
            os.path.join(pasta_landmarks, nome + ".npy"), config)
        linhas.append(resumo_video(nome, ear, estados, config))
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(os.path.join(pasta_saida, f"resumo_{SPLIT.lower()}_p{PROTOCOLO}.csv"), index=False)

    print(f"Protocolo {PROTOCOLO} | {SPLIT}: {len(tabela)} vídeos")
    print(f"Vídeos com algum quadro sem face: {(tabela['quadros_sem_face'] > 0).sum()}")
    print("\nFração de vídeos com pelo menos 1 piscada:")
    for tipo, grupo in tabela.groupby("tipo"):
        texto = f"  {tipo:6s} ({len(grupo)} vídeos): vídeo inteiro {(grupo['piscadas'] > 0).mean():.1%}"
        for tamanho in JANELAS:
            texto += f" | alguma janela de {tamanho} {(grupo[f'janelas_{tamanho}_com_piscada'] > 0).mean():.1%}"
        print(texto)

    duracoes = [d for lista in tabela[tabela["tipo"] == "real"]["duracoes"] for d in lista]
    print(f"\nPiscadas nos vídeos reais: {len(duracoes)}"
          f" | quadros fechados por piscada: mediana {np.median(duracoes):.0f},"
          f" mín {min(duracoes)}, máx {max(duracoes)}")
    for limite in [3, 8, 12]:
        print(f"  com até {limite} quadros fechados: {np.mean(np.array(duracoes) <= limite):.1%}")

    for video in [VIDEO_REAL, VIDEO_PRINT]:
        ear_esq, ear_dir, ear, estados = processar_video(
            os.path.join(pasta_landmarks, video + ".npy"), config)
        pd.DataFrame({"quadro": np.arange(len(ear)), "ear_esq": ear_esq, "ear_dir": ear_dir,
                      "ear": ear, "estado": estados}).to_csv(
            os.path.join(pasta_saida, f"ear_quadros_{video}.csv"), index=False)
        for tamanho in JANELAS:
            janelas = atributos_janelas(ear, estados, tamanho, config["max_quadros_piscada"])
            pd.DataFrame(janelas).to_csv(
                os.path.join(pasta_saida, f"janelas{tamanho}_{video}.csv"), index=False)

    grafico(pasta_landmarks, config, os.path.join(pasta_saida, "ear_real_x_print.png"))
    print("\nArquivos salvos em", pasta_saida)
