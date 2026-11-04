import os
import sys
import time
import shutil
import platform
import subprocess
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (ler_config, carregar_dlib, detectar_face, recortar_face, obter_landmarks,
                   normalizar_entrada, criar_modelo, calcular_ear, achar_piscadas, PASTA_PROJETO)

MODELO = os.path.join(PASTA_PROJETO, "modelos", "espacial_p3_f5.pt")
PASTA_VIDEOS = os.path.join(PASTA_PROJETO, "pi", "videos")
PASTA_SAIDA = os.path.join(PASTA_PROJETO, "resultados", "pi")
VIDEOS = ["5_1_37_1", "5_1_51_1", "5_1_44_2", "5_1_44_3", "5_1_53_4", "5_1_37_5",
          "5_2_40_1", "5_2_49_1", "5_2_46_2", "5_2_55_3", "5_2_50_4", "5_2_51_5",
          "5_3_49_1", "5_3_51_1", "5_3_38_2", "5_3_52_3", "5_3_45_4", "5_3_46_5"]
AQUECIMENTO = 10
REPETICOES = 3
JANELA = 10
QUADROS_DECISAO = 151
FPS_VIDEO = 30
COMPONENTES = ["deteccao", "recorte", "rede", "landmarks", "ear_regra"]


def ler_arquivo(caminho, divisor):
    if not os.path.isfile(caminho):
        return np.nan
    return float(open(caminho).read().split()[0]) / divisor


def temperatura():
    return ler_arquivo("/sys/class/thermal/thermal_zone0/temp", 1000)


def clock_mhz():
    return ler_arquivo("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000)


def memoria_mb(campo="VmRSS"):
    if not os.path.isfile("/proc/self/status"):
        return np.nan
    for linha in open("/proc/self/status"):
        if linha.startswith(campo):
            return float(linha.split()[1]) / 1024
    return np.nan


def ler_quadros(video):
    captura = cv2.VideoCapture(os.path.join(PASTA_VIDEOS, video + ".avi"))
    quadros = []
    while True:
        ok, quadro = captura.read()
        if not ok:
            break
        quadros.append(quadro)
    captura.release()
    return quadros


def processar_video(video, modelos, config, limite=None):
    captura = cv2.VideoCapture(os.path.join(PASTA_VIDEOS, video + ".avi"))
    historico, estados, linhas = [], [], []
    while limite is None or len(linhas) < limite:
        ok, quadro = captura.read()
        if not ok:
            break
        tempos = processar_quadro(quadro, historico, estados, modelos, config)
        linhas.append({"video": video, "quadro": len(linhas), **tempos, "memoria_mb": memoria_mb(),
                       "temperatura_c": temperatura(), "clock_mhz": clock_mhz()})
    captura.release()
    return linhas


def processar_quadro(quadro, historico, estados, modelos, config):
    detector, preditor, modelo = modelos
    tempos = {}
    t = time.perf_counter()
    caixa = detectar_face(quadro, detector, config["lado_maior_deteccao"])
    tempos["deteccao"] = time.perf_counter() - t
    if caixa is None:
        return tempos

    t = time.perf_counter()
    x = normalizar_entrada(recortar_face(quadro, caixa, config["margem_face"], config["tamanho_recorte"]))
    tempos["recorte"] = time.perf_counter() - t

    t = time.perf_counter()
    with torch.no_grad():
        torch.sigmoid(modelo(torch.from_numpy(x).unsqueeze(0)))
    tempos["rede"] = time.perf_counter() - t

    t = time.perf_counter()
    pontos = obter_landmarks(quadro, caixa, preditor)
    tempos["landmarks"] = time.perf_counter() - t

    t = time.perf_counter()
    ear = calcular_ear(pontos[np.newaxis].astype(np.float32))[2][0]
    historico.append(ear)
    estados.append(1 if ear < config["fator_olho_fechado"] * np.median(historico) else 0)
    achar_piscadas(np.array(estados[-JANELA:]), config["max_quadros_piscada"])
    tempos["ear_regra"] = time.perf_counter() - t
    return tempos


def resumir(tabela, config):
    ms = tabela[COMPONENTES] * 1000
    ms["ramo_a"] = ms["deteccao"] + ms["recorte"] + ms["rede"]
    ms["ramo_b"] = ms["deteccao"] + ms["landmarks"] + ms["ear_regra"]
    ms["a_mais_b"] = ms[COMPONENTES].sum(axis=1)
    linhas = []
    for coluna in COMPONENTES + ["ramo_a", "ramo_b", "a_mais_b"]:
        v = ms[coluna].dropna()
        linhas.append({"medida": coluna, "mediana_ms": v.median(), "p90_ms": v.quantile(0.9),
                       "media_ms": v.mean(), "quadros_por_segundo": 1000 / v.mean()})
    resumo = pd.DataFrame(linhas).set_index("medida")

    quadros_a = int(np.ceil(QUADROS_DECISAO / config["passo_quadros"]))
    media = resumo["media_ms"]
    decisoes = {"so_a": quadros_a * media["ramo_a"],
                "so_b": QUADROS_DECISAO * media["ramo_b"],
                "a_mais_b": QUADROS_DECISAO * media["ramo_b"] + quadros_a * (media["recorte"] + media["rede"])}
    return resumo, decisoes, quadros_a


if __name__ == "__main__":
    faltando = [v for v in VIDEOS if not os.path.isfile(os.path.join(PASTA_VIDEOS, v + ".avi"))]
    if faltando:
        sys.exit(f"Faltam vídeos em {PASTA_VIDEOS}: {faltando}")
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    config = ler_config()
    memoria = {"inicio": memoria_mb()}
    modelo = criar_modelo(pre_treinado=False)
    modelo.load_state_dict(torch.load(MODELO, map_location="cpu"))
    modelo.eval()
    memoria["com_mobilenet"] = memoria_mb()
    detector, preditor = carregar_dlib(config)
    memoria["com_dlib"] = memoria_mb()
    modelos = (detector, preditor, modelo)
    print(f"Máquina: {platform.machine()} | Python {platform.python_version()} | PyTorch {torch.__version__}"
          f" | threads {torch.get_num_threads()} | vídeos {len(VIDEOS)}")

    processar_video(VIDEOS[0], modelos, config, AQUECIMENTO)
    linhas = []
    for repeticao in range(1, REPETICOES + 1):
        for video in VIDEOS:
            linhas += [{"repeticao": repeticao, **l} for l in processar_video(video, modelos, config)]
        print(f"Repetição {repeticao}/{REPETICOES} | temperatura {temperatura():.1f} °C | clock {clock_mhz():.0f} MHz")
    tabela = pd.DataFrame(linhas)
    memoria["processando_media"] = tabela["memoria_mb"].mean()
    memoria["processando_pico"] = memoria_mb("VmHWM")

    antes = memoria_mb()
    buffer = ler_quadros(VIDEOS[0])
    memoria["buffer_video_mb"] = memoria_mb() - antes
    memoria["buffer_quadros"] = len(buffer)
    memoria["pico_com_buffer"] = memoria_mb("VmHWM")
    del buffer

    tabela.to_csv(os.path.join(PASTA_SAIDA, "medicoes_quadros.csv"), index=False)
    resumo, decisoes, quadros_a = resumir(tabela, config)
    resumo.to_csv(os.path.join(PASTA_SAIDA, "medicoes_resumo.csv"))
    pd.DataFrame([{"estrategia": k, "processamento_s": v / 1000, "observacao_s": QUADROS_DECISAO / FPS_VIDEO}
                  for k, v in decisoes.items()]).to_csv(os.path.join(PASTA_SAIDA, "medicoes_decisao.csv"), index=False)
    pd.DataFrame([memoria]).to_csv(os.path.join(PASTA_SAIDA, "medicoes_memoria.csv"), index=False)

    print("\nLatência por quadro (ms):")
    print(resumo.round(2).to_string())
    print(f"\nPor decisão ({QUADROS_DECISAO} quadros observados = {QUADROS_DECISAO / FPS_VIDEO:.1f} s;"
          f" o ramo A usa {quadros_a} deles):")
    for estrategia, ms in decisoes.items():
        print(f"  {estrategia:9s} processamento {ms / 1000:.2f} s")
    print("\nMemória do processo (MB):", {k: round(v, 1) for k, v in memoria.items()})
    print(f"Temperatura: máx {tabela['temperatura_c'].max():.1f} °C | clock: mín {tabela['clock_mhz'].min():.0f}"
          f" / máx {tabela['clock_mhz'].max():.0f} MHz")
    if shutil.which("vcgencmd"):
        print("vcgencmd get_throttled:", subprocess.run(["vcgencmd", "get_throttled"],
                                                         capture_output=True, text=True).stdout.strip())
    print("Arquivos salvos em", PASTA_SAIDA)
