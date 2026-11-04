import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (ler_config, info_video, carregar_dlib, detectar_face, recortar_face, obter_landmarks,
                   normalizar_entrada, criar_modelo, calcular_ear, estados_olho, atributos_janelas,
                   confianca, PASTA_PROJETO)
from estrategias import carregar, preparar, guiada, limiar_por_orcamento, acer
from medicoes_pi import PASTA_VIDEOS, PASTA_SAIDA, ler_quadros, temperatura

CONJUNTO = "p3_f5"
MODELO = os.path.join(PASTA_PROJETO, "modelos", f"espacial_{CONJUNTO}.pt")
ARQUIVO_VIDEOS = os.path.join(PASTA_SAIDA, f"estrategias_pi_{CONJUNTO}_videos.csv")
FPS = 30
ESTRATEGIAS = ["só A", "só B", "A+B sempre", "guiada"]


def parametros(config):
    dev, teste = carregar(config, CONJUNTO)
    limiares = preparar(dev, teste)
    limiar_conf = limiar_por_orcamento(dev, config["orcamento_consulta"])
    teste["dec_guiada"] = guiada(teste, limiar_conf)[0]
    temporal = pd.read_csv(os.path.join(PASTA_PROJETO, config["resultados_dir"], "temporal", "resumo.csv"))
    janela = int(temporal.loc[temporal["conjunto"] == CONJUNTO, "janela"].iloc[0])
    p = {**limiares, "limiar_conf": limiar_conf, "janela": janela,
         "min_b": dev["score_b"].min(), "max_b": dev["score_b"].max()}
    pc = teste.set_index("video")[["dec_a", "dec_b", "dec_ab", "dec_guiada"]]
    pc.columns = ESTRATEGIAS
    return p, pc


def esperar(instante):
    falta = instante - time.perf_counter()
    if falta > 0:
        time.sleep(falta)


def score_quadro(quadro, caixa, modelo, config):
    recorte = recortar_face(quadro, caixa, config["margem_face"], config["tamanho_recorte"])
    _, jpg = cv2.imencode(".jpg", recorte, [cv2.IMWRITE_JPEG_QUALITY, config["qualidade_jpg"]])
    x = normalizar_entrada(cv2.imdecode(jpg, cv2.IMREAD_COLOR))
    with torch.no_grad():
        return torch.sigmoid(modelo(torch.from_numpy(x).unsqueeze(0))).item()


def ear_quadro(quadro, caixa, preditor):
    return calcular_ear(obter_landmarks(quadro, caixa, preditor)[np.newaxis].astype(np.float32))[2][0]


def score_temporal(ear, config, janela):
    estados = estados_olho(ear, config["fator_olho_fechado"])
    janelas = pd.DataFrame(atributos_janelas(ear, estados, janela, config["max_quadros_piscada"]))
    validas = janelas[janelas["valida"] == 1]
    return validas["piscada"].mean() if len(validas) else 0.0


def rodar(quadros, estrategia, modelos, p, config):
    detector, preditor, modelo = modelos
    n, passo = len(quadros), config["passo_quadros"]
    faz_b = estrategia in ["só B", "A+B sempre"]
    faz_a = estrategia != "só B"
    scores, ear, caixas = [], np.full(n, np.nan), {}
    ocupado, inicio = 0.0, time.perf_counter()
    for k, quadro in enumerate(quadros):
        usa_a = faz_a and k % passo == 0
        if not (usa_a or faz_b):
            continue
        esperar(inicio + (k + 1) / FPS)
        t = time.perf_counter()
        caixa = detectar_face(quadro, detector, config["lado_maior_deteccao"])
        caixas[k] = caixa
        if caixa is not None and usa_a:
            scores.append(score_quadro(quadro, caixa, modelo, config))
        if caixa is not None and faz_b:
            ear[k] = ear_quadro(quadro, caixa, preditor)
        ocupado += time.perf_counter() - t
    esperar(inicio + n / FPS)

    t = time.perf_counter()
    score_a = np.mean(scores) if scores else 0.0
    consultou = faz_b or (estrategia == "guiada" and confianca(score_a, p["score_a"]) < p["limiar_conf"])
    if consultou and not faz_b:
        for k, quadro in enumerate(quadros):
            caixa = caixas[k] if k in caixas else detectar_face(quadro, detector, config["lado_maior_deteccao"])
            if caixa is not None:
                ear[k] = ear_quadro(quadro, caixa, preditor)
    if estrategia == "só B":
        score, limiar = score_temporal(ear, config, p["janela"]), p["score_b"]
    elif consultou:
        b = np.clip((score_temporal(ear, config, p["janela"]) - p["min_b"]) / (p["max_b"] - p["min_b"]), 0, 1)
        score, limiar = (score_a + b) / 2, p["score_ab"]
    else:
        score, limiar = score_a, p["score_a"]
    fim = time.perf_counter()
    return {"score": score, "decisao": int(score >= limiar), "consultou_b": int(consultou),
            "quadros_observados": n, "quadros_a": len(scores), "quadros_b": int((~np.isnan(ear)).sum()),
            "observacao_s": n / FPS, "decisao_s": fim - inicio, "resposta_s": fim - inicio - n / FPS,
            "cpu_s": ocupado + fim - t}


def previsto(n, estrategia, consultou, c, passo):
    t = cpu = 0.0
    for k in range(n):
        custo = 0.0
        if estrategia != "só B" and k % passo == 0:
            custo += c["deteccao"] + c["recorte"] + c["rede"]
        if estrategia in ["só B", "A+B sempre"]:
            custo += c["landmarks"] + c["ear_regra"] + (0 if custo else c["deteccao"])
        if custo:
            t = max(t, (k + 1) / FPS) + custo
            cpu += custo
    t = max(t, n / FPS)
    if estrategia == "guiada" and consultou:
        extra = n * (c["landmarks"] + c["ear_regra"]) + (n - len(range(0, n, passo))) * c["deteccao"]
        t, cpu = t + extra, cpu + extra
    return t, cpu


def resumir(tabela):
    linhas = []
    for estrategia in ESTRATEGIAS:
        g = tabela[tabela["estrategia"] == estrategia]
        pi, pc = acer(g["decisao"].values, g), acer(g["decisao_pc"].values, g)
        linhas.append({"estrategia": estrategia, "videos": len(g),
                       "igual_ao_pc": f"{(g['decisao'] == g['decisao_pc']).sum()}/{len(g)}",
                       "apcer_max_pi": pi["apcer_max"], "bpcer_pi": pi["bpcer"], "acer_pi": pi["acer"],
                       "acer_pc": pc["acer"], "consulta_b": g["consultou_b"].mean(),
                       "observacao_s": g["observacao_s"].mean(),
                       "resposta_s": g["resposta_s"].mean(), "resposta_s_p90": g["resposta_s"].quantile(0.9),
                       "resposta_s_prevista": g["resposta_s_prevista"].mean(),
                       "cpu_s": g["cpu_s"].mean(), "cpu_s_previsto": g["cpu_s_previsto"].mean()})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    config = ler_config()
    p, pc = parametros(config)
    custos = pd.read_csv(os.path.join(PASTA_SAIDA, "medicoes_resumo.csv"), index_col="medida")["media_ms"] / 1000
    modelo = criar_modelo(pre_treinado=False)
    modelo.load_state_dict(torch.load(MODELO, map_location="cpu"))
    modelos = (*carregar_dlib(config), modelo.eval())
    print("Parâmetros do", CONJUNTO, "(dev):", {k: round(float(v), 4) for k, v in p.items()})

    feitos = set(pd.read_csv(ARQUIVO_VIDEOS, dtype={"video": str})["video"]) if os.path.isfile(ARQUIVO_VIDEOS) else set()
    pendentes = [v for v in pc.index if v not in feitos and os.path.isfile(os.path.join(PASTA_VIDEOS, v + ".avi"))]
    print(f"Teste do {CONJUNTO}: {len(pc)} vídeos | já feitos {len(feitos)} | pendentes na pasta {len(pendentes)}")

    if pendentes:
        rodar(ler_quadros(pendentes[0])[:30], "A+B sempre", modelos, p, config)
    for i, video in enumerate(pendentes, 1):
        quadros = ler_quadros(video)
        linhas = []
        for estrategia in ESTRATEGIAS:
            r = rodar(quadros, estrategia, modelos, p, config)
            decisao_prev, cpu_prev = previsto(len(quadros), estrategia, r["consultou_b"], custos,
                                              config["passo_quadros"])
            linhas.append({"video": video, "tipo": info_video(video)["tipo"], "rotulo": info_video(video)["rotulo"],
                           "estrategia": estrategia, **r, "decisao_pc": int(pc.loc[video, estrategia]),
                           "resposta_s_prevista": decisao_prev - r["observacao_s"], "cpu_s_previsto": cpu_prev,
                           "temperatura_c": temperatura()})
        pd.DataFrame(linhas).to_csv(ARQUIVO_VIDEOS, mode="a", header=not os.path.isfile(ARQUIVO_VIDEOS), index=False)
        print(f"[{i:3d}/{len(pendentes)}] {video}: {len(quadros)} quadros | guiada consultou B: "
              f"{'sim' if linhas[-1]['consultou_b'] else 'não'} | temperatura {temperatura():.1f} °C")
        del quadros

    tabela = pd.read_csv(ARQUIVO_VIDEOS, dtype={"video": str})
    resumo = resumir(tabela)
    resumo.to_csv(os.path.join(PASTA_SAIDA, f"estrategias_pi_{CONJUNTO}_resumo.csv"), index=False)
    print(f"\nResumo ({tabela['video'].nunique()} de {len(pc)} vídeos; métricas em fração, tempos em s):")
    print(resumo.round(3).to_string(index=False))
    print("\nresposta_s = tempo entre o fim do vídeo (último quadro) e a decisão; "
          "*_prevista / *_previsto = calculado com as médias da etapa 13")
    print("Arquivos salvos em", PASTA_SAIDA)
