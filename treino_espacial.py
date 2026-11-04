import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import (ler_config, ler_protocolo, info_video, normalizar_entrada,
                   criar_modelo, calcular_eer, PASTA_PROJETO)

CONJUNTOS = ([(1, None), (2, None)] + [(3, fold) for fold in range(1, 7)]
             + [(4, fold) for fold in range(1, 7)])
MODO_TESTE = False

EPOCAS = 10
TAMANHO_LOTE = 64
TAXA_APRENDIZADO = 1e-4
N_WORKERS = 4
SEMENTE = 42


class DatasetRecortes(Dataset):
    def __init__(self, nomes, pasta_recortes, passo):
        self.itens = []
        for i, nome in enumerate(nomes):
            rotulo = info_video(nome)["rotulo"]
            pasta = os.path.join(pasta_recortes, nome)
            for arquivo in sorted(os.listdir(pasta)):
                if int(arquivo[:-4]) % passo == 0:
                    self.itens.append((os.path.join(pasta, arquivo), rotulo, i))

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, indice):
        caminho, rotulo, video = self.itens[indice]
        img = cv2.imread(caminho)
        x = torch.from_numpy(normalizar_entrada(img))
        return x, torch.tensor(rotulo, dtype=torch.float32), video


def treinar_uma_epoca(modelo, carregador, criterio, otimizador, agendador, dispositivo):
    modelo.train()
    soma_perda = 0
    for x, y, _ in carregador:
        x, y = x.to(dispositivo), y.to(dispositivo)
        saida = modelo(x).squeeze(1)
        perda = criterio(saida, y)
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
        agendador.step()
        soma_perda += perda.item() * len(y)
    return soma_perda / len(carregador.dataset)


def avaliar_dev(modelo, carregador, nomes, dispositivo):
    modelo.eval()
    scores_quadros = [[] for _ in nomes]
    with torch.no_grad():
        for x, _, videos in carregador:
            scores = torch.sigmoid(modelo(x.to(dispositivo)).squeeze(1)).cpu().numpy()
            for s, v in zip(scores, videos.numpy()):
                scores_quadros[v].append(s)
    scores_video = [np.mean(s) for s in scores_quadros if len(s) > 0]
    rotulos_video = [info_video(n)["rotulo"] for n, s in zip(nomes, scores_quadros) if len(s) > 0]
    eer, limiar = calcular_eer(scores_video, rotulos_video)
    return eer, limiar, len(nomes) - len(scores_video)


def ja_treinado(caminho_modelo, caminho_log, epocas):
    if not (os.path.isfile(caminho_modelo) and os.path.isfile(caminho_log)):
        return False
    return len(pd.read_csv(caminho_log)) >= epocas


def treinar_modelo(config, protocolo, fold, dispositivo):
    nome = f"espacial_p{protocolo}" + (f"_f{fold}" if fold else "") + ("_teste" if MODO_TESTE else "")
    caminho_modelo = os.path.join(PASTA_PROJETO, "modelos", nome + ".pt")
    caminho_log = os.path.join(PASTA_PROJETO, config["resultados_dir"], "treino", nome + "_log.csv")
    epocas = 1 if MODO_TESTE else EPOCAS
    print(f"\n========== {nome} ==========")
    if ja_treinado(caminho_modelo, caminho_log, epocas):
        print("Já treinado, pulando")
        return {"modelo": nome, "melhor_eer_dev": pd.read_csv(caminho_log)["eer_dev"].min()}

    torch.manual_seed(SEMENTE)
    pasta_recortes = os.path.join(config["preproc_dir"], "recortes")
    nomes_treino, _ = ler_protocolo(config, protocolo, "Train", fold)
    nomes_dev, _ = ler_protocolo(config, protocolo, "Dev", fold)
    if MODO_TESTE:
        nomes_treino, nomes_dev = nomes_treino[:50], nomes_dev[:50]

    dados_treino = DatasetRecortes(nomes_treino, pasta_recortes, config["passo_quadros"])
    dados_dev = DatasetRecortes(nomes_dev, pasta_recortes, config["passo_quadros"])
    carregador_treino = DataLoader(dados_treino, TAMANHO_LOTE, shuffle=True,
                                   num_workers=N_WORKERS, persistent_workers=True)
    carregador_dev = DataLoader(dados_dev, TAMANHO_LOTE, shuffle=False,
                                num_workers=N_WORKERS, persistent_workers=True)

    n_reais = sum(item[1] for item in dados_treino.itens)
    n_ataques = len(dados_treino) - n_reais
    peso_real = n_ataques / n_reais
    print(f"Treino: {len(nomes_treino)} vídeos, {len(dados_treino)} recortes"
          f" ({n_reais} reais, {n_ataques} ataques) -> peso da classe real {peso_real:.2f}")
    print(f"Dev: {len(nomes_dev)} vídeos, {len(dados_dev)} recortes")

    modelo = criar_modelo().to(dispositivo)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_real], device=dispositivo))
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)
    total_passos = epocas * len(carregador_treino)
    agendador = torch.optim.lr_scheduler.LambdaLR(otimizador, lambda passo: 1 - passo / total_passos)

    melhor_eer = 1.0
    historico = []
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        taxa = otimizador.param_groups[0]["lr"]
        perda = treinar_uma_epoca(modelo, carregador_treino, criterio, otimizador, agendador, dispositivo)
        eer, limiar, sem_face = avaliar_dev(modelo, carregador_dev, nomes_dev, dispositivo)
        marca = ""
        if eer < melhor_eer:
            melhor_eer = eer
            torch.save(modelo.state_dict(), caminho_modelo)
            marca = " <- melhor, salvo"
        print(f"Época {epoca:2d} | taxa no início {taxa:.2e} | perda treino {perda:.4f} | EER dev {eer:.2%}"
              f" (limiar {limiar:.3f}) | {time.time() - inicio:.0f} s{marca}")
        historico.append({"epoca": epoca, "taxa_inicio": taxa, "perda_treino": perda, "eer_dev": eer,
                          "limiar_dev": limiar})
        pd.DataFrame(historico).to_csv(caminho_log, index=False)

    print(f"Vídeos do dev sem recorte (fora do EER): {sem_face}")
    print(f"Melhor EER do dev: {melhor_eer:.2%} | modelo em {caminho_modelo}")
    return {"modelo": nome, "melhor_eer_dev": melhor_eer}


if __name__ == "__main__":
    config = ler_config()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(PASTA_PROJETO, "modelos"), exist_ok=True)
    os.makedirs(os.path.join(PASTA_PROJETO, config["resultados_dir"], "treino"), exist_ok=True)
    conjuntos = CONJUNTOS[:1] if MODO_TESTE else CONJUNTOS
    print(f"Dispositivo: {dispositivo} | modelos a treinar: {len(conjuntos)} | épocas: {EPOCAS}")

    inicio = time.time()
    resumo = []
    for protocolo, fold in conjuntos:
        resumo.append(treinar_modelo(config, protocolo, fold, dispositivo))
        print(f"Tempo total até agora: {(time.time() - inicio) / 60:.0f} min")

    resumo = pd.DataFrame(resumo)
    if not MODO_TESTE:
        resumo.to_csv(os.path.join(PASTA_PROJETO, config["resultados_dir"], "treino", "resumo.csv"), index=False)
    print("\nResumo:")
    print(resumo.to_string(index=False))
