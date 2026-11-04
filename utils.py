import os
import yaml
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import roc_curve, roc_auc_score

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))

TIPOS = {1: "real", 2: "print", 3: "print", 4: "replay", 5: "replay"}

TIPOS_DA_LISTA = {"+1": ["real"], "-1": ["print", "replay"], "-2": ["replay"]}


def ler_config(caminho=None):
    if caminho is None:
        caminho = os.path.join(PASTA_PROJETO, "config.yaml")
    with open(caminho, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def info_video(nome):
    celular, sessao, usuario, arquivo = [int(parte) for parte in nome.split("_")]
    tipo = TIPOS[arquivo]
    rotulo = 1 if tipo == "real" else 0
    return {"celular": celular, "sessao": sessao, "usuario": usuario,
            "arquivo": arquivo, "rotulo": rotulo, "tipo": tipo}


def pasta_do_usuario(usuario):
    if usuario <= 20:
        return "Train_files"
    elif usuario <= 35:
        return "Dev_files"
    return "Test_files"


def caminho_video(config, nome):
    pasta = pasta_do_usuario(info_video(nome)["usuario"])
    return os.path.join(config["dataset_dir"], pasta, nome + ".avi")


def ler_protocolo(config, protocolo, split, fold=None):
    if protocolo in [3, 4] and fold is None:
        raise ValueError(f"O protocolo {protocolo} tem 6 folds: escolha FOLD de 1 a 6")
    if fold is None:
        nome_lista = f"{split}.txt"
    else:
        nome_lista = f"{split}_{fold}.txt"
    caminho_lista = os.path.join(config["dataset_dir"], "Protocols",
                                 f"Protocol_{protocolo}", nome_lista)

    nomes = []
    caminhos = []
    with open(caminho_lista, "r") as f:
        for linha in f:
            linha = linha.strip()
            if linha == "":
                continue
            rotulo_lista, nome = linha.split(",")
            info = info_video(nome)

            aceitos = TIPOS_DA_LISTA[rotulo_lista]
            if split == "Test" and rotulo_lista == "-1":
                aceitos = ["print"]
            if info["tipo"] not in aceitos:
                print("  AVISO: rótulo da lista não bate com o nome:", linha)

            nomes.append(nome)
            caminhos.append(caminho_video(config, nome))
    return nomes, caminhos


def calcular_apcer_bpcer(scores, rotulos, tipos, limiar):
    scores = np.array(scores)
    rotulos = np.array(rotulos)
    tipos = np.array(tipos)
    aceito = scores >= limiar

    resultado = {}
    tipos_ataque = sorted(set(tipos[rotulos == 0]))
    for tipo in tipos_ataque:
        ataques_do_tipo = (rotulos == 0) & (tipos == tipo)
        resultado["apcer_" + tipo] = aceito[ataques_do_tipo].mean()
    resultado["apcer_max"] = max(resultado["apcer_" + t] for t in tipos_ataque)

    resultado["bpcer"] = 1 - aceito[rotulos == 1].mean()
    resultado["acer"] = (resultado["apcer_max"] + resultado["bpcer"]) / 2
    return resultado


def calcular_eer(scores, rotulos):
    fpr, tpr, limiares = roc_curve(rotulos, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    eer = (fpr[i] + fnr[i]) / 2
    return eer, limiares[i]


def calcular_auc(scores, rotulos):
    return roc_auc_score(rotulos, scores)


def avaliar_dev_teste(scores_dev, rotulos_dev, scores_teste, rotulos_teste, tipos_teste):
    eer_dev, limiar = calcular_eer(scores_dev, rotulos_dev)
    resultado = calcular_apcer_bpcer(scores_teste, rotulos_teste, tipos_teste, limiar)
    eer_teste, _ = calcular_eer(scores_teste, rotulos_teste)
    resultado["limiar_dev"] = limiar
    resultado["eer_dev"] = eer_dev
    resultado["eer_teste"] = eer_teste
    resultado["auc_teste"] = calcular_auc(scores_teste, rotulos_teste)
    return resultado


OLHO_ESQUERDO = list(range(36, 42))
OLHO_DIREITO = list(range(42, 48))

MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def carregar_dlib(config):
    detector = dlib.get_frontal_face_detector()
    caminho = config["preditor_landmarks"]
    if not os.path.isabs(caminho):
        caminho = os.path.join(PASTA_PROJETO, caminho)
    preditor = dlib.shape_predictor(caminho)
    return detector, preditor


def detectar_face(quadro, detector, lado_maior=640):
    altura, largura = quadro.shape[:2]
    escala = lado_maior / max(altura, largura)
    pequeno = cv2.resize(quadro, (round(largura * escala), round(altura * escala)),
                         interpolation=cv2.INTER_AREA)
    cinza = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)
    faces = detector(cinza, 0)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f.width() * f.height())
    return (int(face.left() / escala), int(face.top() / escala),
            int(face.right() / escala), int(face.bottom() / escala))


def quadrado_com_margem(caixa, margem=1.4):
    x1, y1, x2, y2 = caixa
    lado = int(max(x2 - x1, y2 - y1) * margem)
    qx = int((x1 + x2) / 2 - lado / 2)
    qy = int((y1 + y2) / 2 - lado / 2)
    return qx, qy, lado


def recortar_face(quadro, caixa, margem=1.4, tamanho=224):
    qx, qy, lado = quadrado_com_margem(caixa, margem)
    altura, largura = quadro.shape[:2]
    recorte = np.zeros((lado, lado, 3), dtype=quadro.dtype)
    x1, y1 = max(qx, 0), max(qy, 0)
    x2, y2 = min(qx + lado, largura), min(qy + lado, altura)
    if x2 > x1 and y2 > y1:
        recorte[y1 - qy:y2 - qy, x1 - qx:x2 - qx] = quadro[y1:y2, x1:x2]
    return cv2.resize(recorte, (tamanho, tamanho), interpolation=cv2.INTER_AREA)


def obter_landmarks(quadro, caixa, preditor):
    cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
    retangulo = dlib.rectangle(caixa[0], caixa[1], caixa[2], caixa[3])
    forma = preditor(cinza, retangulo)
    return np.array([[forma.part(i).x, forma.part(i).y] for i in range(68)])


def centros_olhos(pontos):
    return pontos[OLHO_ESQUERDO].mean(axis=0), pontos[OLHO_DIREITO].mean(axis=0)


def ler_olhos_txt(caminho_avi):
    olhos = {}
    with open(caminho_avi.replace(".avi", ".txt"), "r") as f:
        for linha in f:
            valores = [float(v) for v in linha.strip().split(",") if v.strip() != ""]
            if len(valores) != 5 or sum(valores[1:]) == 0:
                continue
            olhos[int(valores[0])] = valores[1:]
    return olhos


def normalizar_entrada(recorte_bgr):
    rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEDIA_IMAGENET) / DESVIO_IMAGENET
    return rgb.transpose(2, 0, 1)


def criar_modelo(pre_treinado=True, neuronios_ocultos=None):
    pesos = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pre_treinado else None
    modelo = models.mobilenet_v3_small(weights=pesos)
    if neuronios_ocultos is None:
        modelo.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(576, 1))
    else:
        modelo.classifier = nn.Sequential(
            nn.Linear(576, neuronios_ocultos),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(neuronios_ocultos, 1),
        )
    return modelo


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def confianca(score, limiar):
    return abs(logit(score) - logit(limiar))


def ear_olho(pontos):
    vertical_1 = np.linalg.norm(pontos[:, 1] - pontos[:, 5], axis=1)
    vertical_2 = np.linalg.norm(pontos[:, 2] - pontos[:, 4], axis=1)
    horizontal = np.linalg.norm(pontos[:, 0] - pontos[:, 3], axis=1)
    return (vertical_1 + vertical_2) / (2 * horizontal)


def calcular_ear(landmarks):
    ear_esq = ear_olho(landmarks[:, OLHO_ESQUERDO])
    ear_dir = ear_olho(landmarks[:, OLHO_DIREITO])
    return ear_esq, ear_dir, (ear_esq + ear_dir) / 2


def estados_olho(ear, fator=0.75):
    estados = np.full(len(ear), -1)
    for t in range(len(ear)):
        if np.isnan(ear[t]):
            continue
        referencia = np.nanmedian(ear[:t + 1])
        estados[t] = 1 if ear[t] < fator * referencia else 0
    return estados


def achar_piscadas(estados, max_fechados=12):
    duracoes = []
    t = 1
    while t < len(estados):
        if estados[t] == 1 and estados[t - 1] == 0:
            fim = t
            while fim < len(estados) and estados[fim] == 1:
                fim += 1
            if fim < len(estados) and estados[fim] == 0 and fim - t <= max_fechados:
                duracoes.append(fim - t)
            t = fim
        else:
            t += 1
    return duracoes


def atributos_janelas(ear, estados, tamanho=5, max_fechados=12):
    janelas = []
    for inicio in range(len(ear) - tamanho + 1):
        e = ear[inicio:inicio + tamanho]
        s = estados[inicio:inicio + tamanho]
        valida = not np.isnan(e).any()
        janelas.append({"inicio": inicio, "valida": int(valida),
                        "ear_min": e.min() if valida else np.nan,
                        "ear_variacao": e.max() - e.min() if valida else np.nan,
                        "piscada": int(valida and len(achar_piscadas(s, max_fechados)) > 0),
                        "quadros_fechados": int((s == 1).sum()) if valida else 0})
    return janelas
