import os
import sys
import math
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ler_config, ler_protocolo, criar_modelo, avaliar_dev_teste, PASTA_PROJETO
from treino_espacial import (DatasetRecortes, avaliar_dev, EPOCAS, TAMANHO_LOTE, TAXA_APRENDIZADO,
                             N_WORKERS, SEMENTE)
from avalia_espacial import pontuar_quadros, score_video, listas

PROTOCOLO = 1
DECAIMENTO = "linear"  # "linear" ou "cosseno"
MODO_TESTE = False


def fator(passo, total):
    progresso = passo / total
    return 1 - progresso if DECAIMENTO == "linear" else 0.5 * (1 + math.cos(math.pi * progresso))


def treinar_uma_epoca(modelo, carregador, criterio, otimizador, agendador, dispositivo):
    modelo.train()
    soma_perda = 0
    for x, y, _ in carregador:
        x, y = x.to(dispositivo), y.to(dispositivo)
        perda = criterio(modelo(x).squeeze(1), y)
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
        agendador.step()
        soma_perda += perda.item() * len(y)
    return soma_perda / len(carregador.dataset)


def treinar(config, nomes_treino, nomes_dev, caminhos, caminho_log, epocas, dispositivo):
    torch.manual_seed(SEMENTE)
    pasta_recortes = os.path.join(config["preproc_dir"], "recortes")
    dados_treino = DatasetRecortes(nomes_treino, pasta_recortes, config["passo_quadros"])
    dados_dev = DatasetRecortes(nomes_dev, pasta_recortes, config["passo_quadros"])
    carregador_treino = DataLoader(dados_treino, TAMANHO_LOTE, shuffle=True, num_workers=N_WORKERS,
                                   persistent_workers=True)
    carregador_dev = DataLoader(dados_dev, TAMANHO_LOTE, shuffle=False, num_workers=N_WORKERS,
                                persistent_workers=True)
    n_reais = sum(item[1] for item in dados_treino.itens)
    peso_real = (len(dados_treino) - n_reais) / n_reais

    modelo = criar_modelo().to(dispositivo)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_real], device=dispositivo))
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)
    total = epocas * len(carregador_treino)
    agendador = torch.optim.lr_scheduler.LambdaLR(otimizador, lambda passo: fator(passo, total))

    melhor_eer, historico = 1.0, []
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        taxa = otimizador.param_groups[0]["lr"]
        perda = treinar_uma_epoca(modelo, carregador_treino, criterio, otimizador, agendador, dispositivo)
        eer, limiar, _ = avaliar_dev(modelo, carregador_dev, nomes_dev, dispositivo)
        marca = ""
        if eer < melhor_eer:
            melhor_eer = eer
            torch.save(modelo.state_dict(), caminhos["melhor no dev"])
            marca = " <- melhor, salvo"
        print(f"Época {epoca:2d} | taxa no início {taxa:.2e} | perda treino {perda:.4f} | EER dev {eer:.2%}"
              f" | {time.time() - inicio:.0f} s{marca}")
        historico.append({"epoca": epoca, "taxa_inicio": taxa, "perda_treino": perda, "eer_dev": eer,
                          "limiar_dev": limiar})
        pd.DataFrame(historico).to_csv(caminho_log, index=False)
    torch.save(modelo.state_dict(), caminhos["última época"])


def avaliar(caminho_modelo, config, nomes_dev, nomes_teste, dispositivo):
    modelo = criar_modelo(pre_treinado=False)
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=dispositivo))
    modelo.to(dispositivo).eval()
    pasta_recortes, passo = os.path.join(config["preproc_dir"], "recortes"), config["passo_quadros"]
    dev = score_video(pontuar_quadros(modelo, nomes_dev, pasta_recortes, "dev", dispositivo, passo))
    teste = score_video(pontuar_quadros(modelo, nomes_teste, pasta_recortes, "teste", dispositivo, passo))
    sd, rd, _ = listas(dev, nomes_dev)
    st, rt, tt = listas(teste, nomes_teste)
    return avaliar_dev_teste(sd, rd, st, rt, tt)


if __name__ == "__main__":
    config = ler_config()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    pasta = os.path.join(PASTA_PROJETO, config["resultados_dir"], "teste_decaimento")
    os.makedirs(pasta, exist_ok=True)
    nome = f"espacial_p{PROTOCOLO}_{DECAIMENTO}" + ("_teste" if MODO_TESTE else "")
    caminhos = {"última época": os.path.join(PASTA_PROJETO, "modelos", nome + "_ultima.pt"),
                "melhor no dev": os.path.join(PASTA_PROJETO, "modelos", nome + "_melhor.pt")}
    caminho_log = os.path.join(pasta, nome + "_log.csv")

    nomes_treino, _ = ler_protocolo(config, PROTOCOLO, "Train")
    nomes_dev, _ = ler_protocolo(config, PROTOCOLO, "Dev")
    nomes_teste, _ = ler_protocolo(config, PROTOCOLO, "Test")
    epocas = EPOCAS
    if MODO_TESTE:
        nomes_treino, nomes_dev, nomes_teste, epocas = nomes_treino[:50], nomes_dev[:50], nomes_teste[:50], 2
    print(f"Dispositivo: {dispositivo} | P{PROTOCOLO} | decaimento {DECAIMENTO} | épocas {epocas}")

    if all(os.path.isfile(c) for c in caminhos.values()):
        print("Modelos do teste já existem, pulando o treino")
    else:
        treinar(config, nomes_treino, nomes_dev, caminhos, caminho_log, epocas, dispositivo)

    atual = os.path.join(PASTA_PROJETO, "modelos", f"espacial_p{PROTOCOLO}.pt")
    modelos = {"atual (taxa fixa, melhor no dev)": atual, **{f"{DECAIMENTO}, {k}": c for k, c in caminhos.items()}}
    linhas = [{"modelo": k, **avaliar(c, config, nomes_dev, nomes_teste, dispositivo)} for k, c in modelos.items()]
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(os.path.join(pasta, nome + "_comparacao.csv"), index=False)
    colunas = ["eer_dev", "apcer_max", "bpcer", "acer", "eer_teste", "auc_teste"]
    print("\nComparação (%; AUC em fração). A decisão deve olhar o EER do dev, não o teste:")
    print(pd.concat([tabela["modelo"], (100 * tabela[colunas[:-1]]).round(2), tabela["auc_teste"].round(3)],
                    axis=1).to_string(index=False))
    print("\nArquivos salvos em", pasta)
