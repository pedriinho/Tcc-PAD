import os
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ler_config, ler_protocolo, info_video, PASTA_PROJETO

FOLDS = {1: [None], 2: [None], 3: [1, 2, 3, 4, 5, 6], 4: [1, 2, 3, 4, 5, 6]}
SPLITS = ["Train", "Dev", "Test"]


def contar_videos_nas_pastas(config):
    total = 0
    for pasta in ["Train_files", "Dev_files", "Test_files"]:
        arquivos = os.listdir(os.path.join(config["dataset_dir"], pasta))
        n = len([a for a in arquivos if a.endswith(".avi")])
        print(f"  {pasta:12s} {n} vídeos")
        total += n
    print(f"  Total        {total} vídeos")


def contar_tipos(nomes):
    contagem = {"real": 0, "print": 0, "replay": 0}
    for nome in nomes:
        contagem[info_video(nome)["tipo"]] += 1
    return contagem


def conferir_arquivos(caminhos):
    falta_avi = 0
    falta_txt = 0
    for caminho in caminhos:
        if not os.path.isfile(caminho):
            print("  FALTANDO:", caminho)
            falta_avi += 1
        if not os.path.isfile(caminho.replace(".avi", ".txt")):
            print("  FALTANDO:", caminho.replace(".avi", ".txt"))
            falta_txt += 1
    return falta_avi, falta_txt


def valores(nomes, campo):
    return sorted(set(info_video(nome)[campo] for nome in nomes))


if __name__ == "__main__":
    config = ler_config()
    print("Vídeos nas pastas:")
    contar_videos_nas_pastas(config)

    linhas = []
    for protocolo in [1, 2, 3, 4]:
        print(f"\n===== Protocolo {protocolo} =====")
        for fold in FOLDS[protocolo]:
            for split in SPLITS:
                nomes, caminhos = ler_protocolo(config, protocolo, split, fold)
                contagem = contar_tipos(nomes)
                falta_avi, falta_txt = conferir_arquivos(caminhos)
                usuarios = valores(nomes, "usuario")

                nome_split = split if fold is None else f"{split}_{fold}"
                print(f"{nome_split:8s} total {len(nomes):4d} | real {contagem['real']:3d}"
                      f" | print {contagem['print']:3d} | replay {contagem['replay']:3d}"
                      f" | faltando avi {falta_avi} txt {falta_txt}")
                print(f"         celulares {valores(nomes, 'celular')}"
                      f" sessões {valores(nomes, 'sessao')}"
                      f" arquivos {valores(nomes, 'arquivo')}"
                      f" usuários {usuarios[0]}-{usuarios[-1]}")

                linhas.append({"protocolo": protocolo, "fold": fold, "split": split,
                               "total": len(nomes), "real": contagem["real"],
                               "print": contagem["print"], "replay": contagem["replay"],
                               "faltando_avi": falta_avi, "faltando_txt": falta_txt})

    tabela = pd.DataFrame(linhas)
    saida = os.path.join(PASTA_PROJETO, config["resultados_dir"], "verificacoes", "protocolos.csv")
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    tabela.to_csv(saida, index=False)
    print("\nTabela salva em", saida)
    print("Total de arquivos faltando:", tabela["faltando_avi"].sum() + tabela["faltando_txt"].sum())
