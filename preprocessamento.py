import os
import time
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool

from utils import (ler_config, carregar_dlib, detectar_face, recortar_face,
                   obter_landmarks)

LIMITE_VIDEOS = None
N_PROCESSOS = 6

config = ler_config()
detector = None
preditor = None


def iniciar_processo():
    global detector, preditor
    cv2.setNumThreads(1)
    detector, preditor = carregar_dlib(config)


def listar_videos():
    videos = []
    for pasta in ["Train_files", "Dev_files", "Test_files"]:
        for arquivo in sorted(os.listdir(os.path.join(config["dataset_dir"], pasta))):
            if arquivo.endswith(".avi"):
                videos.append((arquivo[:-4], os.path.join(config["dataset_dir"], pasta, arquivo)))
    return videos


def processar_video(item):
    nome, caminho = item
    inicio = time.time()
    pasta_recortes = os.path.join(config["preproc_dir"], "recortes", nome)
    os.makedirs(pasta_recortes, exist_ok=True)

    captura = cv2.VideoCapture(caminho)
    todos_pontos = []
    n = 0
    com_face = 0
    recortes = 0
    while True:
        ok, quadro = captura.read()
        if not ok:
            break
        caixa = detectar_face(quadro, detector, config["lado_maior_deteccao"])
        if caixa is None:
            todos_pontos.append(np.full((68, 2), np.nan, dtype=np.float32))
        else:
            com_face += 1
            todos_pontos.append(obter_landmarks(quadro, caixa, preditor).astype(np.float32))
            if n % config["passo_quadros"] == 0:
                recorte = recortar_face(quadro, caixa, config["margem_face"],
                                        config["tamanho_recorte"])
                cv2.imwrite(os.path.join(pasta_recortes, f"{n:03d}.jpg"), recorte,
                            [cv2.IMWRITE_JPEG_QUALITY, config["qualidade_jpg"]])
                recortes += 1
        n += 1
    captura.release()

    np.save(os.path.join(config["preproc_dir"], "landmarks", nome + ".npy"), np.stack(todos_pontos))
    tempo = time.time() - inicio
    return {"video": nome, "quadros": n, "quadros_com_face": com_face,
            "taxa_face": round(com_face / n, 4), "recortes_salvos": recortes,
            "tempo_s": round(tempo, 2), "ms_por_quadro": round(1000 * tempo / n, 1)}


def ja_processados(caminho_csv):
    if not os.path.isfile(caminho_csv):
        return set()
    return set(pd.read_csv(caminho_csv, dtype={"video": str})["video"])


if __name__ == "__main__":
    os.makedirs(os.path.join(config["preproc_dir"], "landmarks"), exist_ok=True)
    caminho_csv = os.path.join(config["preproc_dir"], "resumo_preproc.csv")

    videos = listar_videos()
    feitos = ja_processados(caminho_csv)
    pendentes = [v for v in videos if v[0] not in feitos]
    if LIMITE_VIDEOS is not None:
        pendentes = pendentes[:LIMITE_VIDEOS]
    print(f"Vídeos: {len(videos)} | já processados: {len(feitos)} | agora: {len(pendentes)}")

    inicio = time.time()
    with Pool(N_PROCESSOS, initializer=iniciar_processo) as pool:
        for i, linha in enumerate(pool.imap_unordered(processar_video, pendentes), start=1):
            novo = not os.path.isfile(caminho_csv)
            pd.DataFrame([linha]).to_csv(caminho_csv, mode="a", header=novo, index=False)
            decorrido = time.time() - inicio
            restante = decorrido / i * (len(pendentes) - i)
            print(f"[{i}/{len(pendentes)}] {linha['video']} | face {linha['taxa_face']:.0%}"
                  f" | {linha['tempo_s']:.1f} s | faltam ~{restante / 60:.0f} min")

    resumo = pd.read_csv(caminho_csv, dtype={"video": str})
    print(f"\nTotal no CSV: {len(resumo)} vídeos | tempo desta execução: {(time.time() - inicio) / 60:.1f} min")
    print(f"Taxa média de face: {resumo['taxa_face'].mean():.2%}")
    print(f"Vídeos com taxa de face < 90%: {(resumo['taxa_face'] < 0.9).sum()}")
    print(f"Tempo médio por quadro (por processo): {resumo['ms_por_quadro'].mean():.1f} ms")
    print("Saída em", config["preproc_dir"])
