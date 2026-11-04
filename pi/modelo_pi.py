import os
import sys
import platform
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import criar_modelo, normalizar_entrada, PASTA_PROJETO
from medicoes_pi import VIDEOS

CONJUNTO = "p3_f5"
MODELO = os.path.join(PASTA_PROJETO, "modelos", f"espacial_{CONJUNTO}.pt")
SCORES_PC = os.path.join(PASTA_PROJETO, "resultados", "espacial", CONJUNTO, "scores_quadros.csv")
RESUMO_PC = os.path.join(PASTA_PROJETO, "resultados", "espacial", "resumo.csv")
PASTA_AMOSTRAS = os.path.join(PASTA_PROJETO, "pi", "amostras")
PASTA_SAIDA = os.path.join(PASTA_PROJETO, "resultados", "pi")
PASSO = 3


def carregar_modelo():
    modelo = criar_modelo(pre_treinado=False)
    modelo.load_state_dict(torch.load(MODELO, map_location="cpu"))
    return modelo.eval()


def contar_parametros(modelo):
    conta = lambda parte: sum(p.numel() for p in parte.parameters())
    return conta(modelo), conta(modelo.features), conta(modelo.classifier)


def contar_flops(modelo):
    entrada = torch.zeros(1, 3, 224, 224)
    contador = FlopCounterMode(display=False)
    with torch.no_grad(), contador:
        modelo(entrada)
    return contador.get_total_flops()


def scores_video(modelo, video):
    pasta = os.path.join(PASTA_AMOSTRAS, video)
    arquivos = sorted(a for a in os.listdir(pasta) if int(a[:-4]) % PASSO == 0)
    lote = np.stack([normalizar_entrada(cv2.imread(os.path.join(pasta, a))) for a in arquivos])
    with torch.no_grad():
        scores = torch.sigmoid(modelo(torch.from_numpy(lote)).squeeze(1)).numpy()
    return pd.DataFrame({"video": video, "quadro": [int(a[:-4]) for a in arquivos], "score_pi": scores})


if __name__ == "__main__":
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    print(f"Máquina: {platform.machine()} | Python {platform.python_version()} | PyTorch {torch.__version__}"
          f" | threads {torch.get_num_threads()}")

    modelo = carregar_modelo()
    total, backbone, cabeca = contar_parametros(modelo)
    tamanho_mb = os.path.getsize(MODELO) / 1024 ** 2
    flops = contar_flops(modelo)
    info = {"arquivo_mb": tamanho_mb, "parametros": total, "parametros_backbone": backbone,
            "parametros_cabeca": cabeca, "flops": flops, "macs": flops / 2}
    pd.DataFrame([info]).to_csv(os.path.join(PASTA_SAIDA, "modelo_info.csv"), index=False)

    print(f"\nArquivo do modelo: {tamanho_mb:.2f} MB")
    print(f"Parâmetros: {total:,} (backbone {backbone:,} + cabeça {cabeca:,})".replace(",", "."))
    print(f"FLOPs por recorte 224x224: {flops / 1e6:.1f} M ({flops / 2 / 1e6:.1f} M multiplicações-somas)")

    pc = pd.read_csv(SCORES_PC, dtype={"video": str})
    pc = pc[pc["split"] == "teste"].rename(columns={"score": "score_pc"})
    comparacao = pd.concat([scores_video(modelo, v) for v in VIDEOS]).merge(pc, on=["video", "quadro"])
    comparacao["diferenca"] = (comparacao["score_pi"] - comparacao["score_pc"]).abs()
    comparacao.to_csv(os.path.join(PASTA_SAIDA, "comparacao_scores.csv"), index=False)

    resumo = pd.read_csv(RESUMO_PC)
    limiar = resumo[resumo["modelo"] == f"espacial_{CONJUNTO}"]["limiar_dev"].iloc[0]
    print(f"\nScore deste computador x score do PC (mesmos recortes, limiar do {CONJUNTO} = {limiar:.4f}):")
    iguais = 0
    for video, g in comparacao.groupby("video"):
        aqui, no_pc = g["score_pi"].mean(), g["score_pc"].mean()
        mesma = (aqui >= limiar) == (no_pc >= limiar)
        iguais += mesma
        print(f"  {video}: {len(g)} recortes | score do vídeo aqui {aqui:.6f} | no PC {no_pc:.6f}"
              f" | maior diferença por quadro {g['diferenca'].max():.2e} | mesma decisão: {'sim' if mesma else 'NÃO'}")
    print(f"\nDiferença média por quadro: {comparacao['diferenca'].mean():.2e}"
          f" | vídeos com a mesma decisão: {iguais}/{len(VIDEOS)}")
    print("Obs.: o PC calculou com TF32 na GPU (menos precisão); aqui o cálculo é em float32 completo.")
    print("Arquivos salvos em", PASTA_SAIDA)
