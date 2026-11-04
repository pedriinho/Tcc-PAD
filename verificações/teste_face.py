import os
import sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (ler_config, caminho_video, carregar_dlib, detectar_face,
                   quadrado_com_margem, recortar_face, obter_landmarks,
                   centros_olhos, ler_olhos_txt, normalizar_entrada,
                   OLHO_ESQUERDO, OLHO_DIREITO, PASTA_PROJETO)

VIDEO = "1_1_01_1"
QUADROS_SALVAR = [0, 50, 100, 150]


def desenhar(quadro, caixa, pontos, margem):
    img = quadro.copy()
    cv2.rectangle(img, (caixa[0], caixa[1]), (caixa[2], caixa[3]), (0, 255, 0), 4)
    qx, qy, lado = quadrado_com_margem(caixa, margem)
    cv2.rectangle(img, (qx, qy), (qx + lado, qy + lado), (0, 255, 255), 4)
    for i, (x, y) in enumerate(pontos):
        cor = (255, 0, 0) if i in OLHO_ESQUERDO + OLHO_DIREITO else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), 5, cor, -1)
    return img


def salvar_imagens(pasta, n, quadro, caixa, pontos, recorte, olhos_txt, margem):
    img = desenhar(quadro, caixa, pontos, margem)
    if n in olhos_txt:
        xe, ye, xd, yd = olhos_txt[n]
        for x, y in [(xe, ye), (xd, yd)]:
            cv2.drawMarker(img, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imwrite(os.path.join(pasta, f"quadro{n:03d}_anotado.jpg"), img)
    cv2.imwrite(os.path.join(pasta, f"quadro{n:03d}_recorte.jpg"), recorte)


if __name__ == "__main__":
    config = ler_config()
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    caminho = caminho_video(config, video)
    pasta_saida = os.path.join(PASTA_PROJETO, config["resultados_dir"], "verificacoes", "face", video)
    os.makedirs(pasta_saida, exist_ok=True)

    detector, preditor = carregar_dlib(config)
    olhos_txt = ler_olhos_txt(caminho)
    margem = config["margem_face"]

    captura = cv2.VideoCapture(caminho)
    fps = captura.get(cv2.CAP_PROP_FPS)
    n = 0
    com_face = 0
    dist_esq, dist_dir, interocular = [], [], []
    while True:
        ok, quadro = captura.read()
        if not ok:
            break
        if n == 0:
            altura, largura = quadro.shape[:2]
        caixa = detectar_face(quadro, detector, config["lado_maior_deteccao"])
        if caixa is not None:
            com_face += 1
            pontos = obter_landmarks(quadro, caixa, preditor)
            centro_esq, centro_dir = centros_olhos(pontos)
            if n in olhos_txt:
                xe, ye, xd, yd = olhos_txt[n]
                dist_esq.append(np.hypot(centro_esq[0] - xe, centro_esq[1] - ye))
                dist_dir.append(np.hypot(centro_dir[0] - xd, centro_dir[1] - yd))
                interocular.append(np.hypot(xd - xe, yd - ye))
            if n in QUADROS_SALVAR:
                recorte = recortar_face(quadro, caixa, margem, config["tamanho_recorte"])
                salvar_imagens(pasta_saida, n, quadro, caixa, pontos, recorte, olhos_txt, margem)
        n += 1
    captura.release()

    print("Vídeo:", video)
    print(f"FPS: {fps:.2f} | resolução (largura x altura): {largura} x {altura} | quadros: {n}")
    print(f"Quadros com face (dlib): {com_face}/{n} = {com_face / n:.1%}")
    print(f"Quadros válidos no .txt oficial: {len(olhos_txt)}/{n}")

    print("\nDistância entre os centros dos olhos (dlib x .txt oficial), em pixels:")
    print(f"  olho esquerdo: média {np.mean(dist_esq):.1f} | máxima {np.max(dist_esq):.1f}")
    print(f"  olho direito : média {np.mean(dist_dir):.1f} | máxima {np.max(dist_dir):.1f}")
    media = np.mean(dist_esq + dist_dir)
    print(f"  geral        : média {media:.1f} px"
          f" (distância entre os olhos ~{np.mean(interocular):.0f} px,"
          f" ou seja {media / np.mean(interocular):.1%} dela)")

    entrada = normalizar_entrada(recorte)
    print(f"\nEntrada da rede: formato {entrada.shape}, tipo {entrada.dtype},"
          f" mín {entrada.min():.2f}, máx {entrada.max():.2f}")
    print("Imagens salvas em", pasta_saida)
