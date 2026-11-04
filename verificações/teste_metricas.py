import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (info_video, calcular_eer, calcular_apcer_bpcer,
                   calcular_auc, avaliar_dev_teste)

dev_scores = [0.90, 0.80, 0.70, 0.40, 0.60, 0.30, 0.20, 0.10]
dev_rotulos = [1, 1, 1, 1, 0, 0, 0, 0]

teste_nomes = ["1_3_36_1", "1_3_37_1", "1_3_38_1", "1_3_39_1",
               "1_3_36_2", "1_3_37_3", "1_3_36_4", "1_3_37_5"]
teste_scores = [0.95, 0.75, 0.62, 0.55, 0.65, 0.20, 0.40, 0.30]
teste_rotulos = [info_video(n)["rotulo"] for n in teste_nomes]
teste_tipos = [info_video(n)["tipo"] for n in teste_nomes]


def mostrar_tabela_dev():
    print("PASSO 1 - escolher o limiar no dev (EER)")
    print("Cada score do dev é testado como limiar (real se score >= limiar)")
    reais = [s for s, r in zip(dev_scores, dev_rotulos) if r == 1]
    ataques = [s for s, r in zip(dev_scores, dev_rotulos) if r == 0]
    print(f"{'limiar':>7s} | {'ataques aceitos':>15s} | {'reais rejeitados':>16s} | {'diferença':>9s}")
    for limiar in sorted(dev_scores, reverse=True):
        aceitos = sum(s >= limiar for s in ataques)
        rejeitados = sum(s < limiar for s in reais)
        apcer = aceitos / len(ataques)
        bpcer = rejeitados / len(reais)
        print(f"{limiar:7.2f} | {aceitos}/{len(ataques)} = {apcer:5.2f}"
              f"     | {rejeitados}/{len(reais)} = {bpcer:5.2f}      | {abs(apcer - bpcer):9.2f}")
    eer, limiar = calcular_eer(dev_scores, dev_rotulos)
    print(f"-> menor diferença: limiar = {limiar:.2f}, EER do dev = {eer:.2%}\n")
    return limiar


def mostrar_decisoes_teste(limiar):
    print(f"PASSO 2 - aplicar o limiar {limiar:.2f} ao teste")
    print(f"{'vídeo':>9s} | {'tipo':>6s} | {'score':>5s} | {'decisão':>7s} | erro?")
    for nome, tipo, score in zip(teste_nomes, teste_tipos, teste_scores):
        decisao = "real" if score >= limiar else "ataque"
        verdade = "real" if tipo == "real" else "ataque"
        erro = "ERRO" if decisao != verdade else ""
        print(f"{nome:>9s} | {tipo:>6s} | {score:5.2f} | {decisao:>7s} | {erro}")
    print()


def mostrar_contas(limiar):
    print("PASSO 3 - contas no teste")
    for tipo in ["print", "replay"]:
        scores_tipo = [s for s, t in zip(teste_scores, teste_tipos) if t == tipo]
        aceitos = sum(s >= limiar for s in scores_tipo)
        print(f"APCER {tipo:6s} = ataques {tipo} aceitos / total {tipo} = {aceitos}/{len(scores_tipo)}"
              f" = {aceitos / len(scores_tipo):.2%}")
    reais = [s for s, t in zip(teste_scores, teste_tipos) if t == "real"]
    rejeitados = sum(s < limiar for s in reais)
    print(f"BPCER        = reais rejeitados / total reais = {rejeitados}/{len(reais)}"
          f" = {rejeitados / len(reais):.2%}")
    m = calcular_apcer_bpcer(teste_scores, teste_rotulos, teste_tipos, limiar)
    print(f"APCER (pior) = max(print, replay) = {m['apcer_max']:.2%}")
    print(f"ACER         = (APCER pior + BPCER) / 2 = ({m['apcer_max']:.2%} + {m['bpcer']:.2%}) / 2"
          f" = {m['acer']:.2%}\n")


def mostrar_eer_auc_teste():
    print("PASSO 4 - EER e AUC do teste (não usam o limiar do dev)")
    eer, limiar_teste = calcular_eer(teste_scores, teste_rotulos)
    print(f"EER do teste = {eer:.2%} (no limiar {limiar_teste:.2f}, só informativo)")
    reais = [s for s, r in zip(teste_scores, teste_rotulos) if r == 1]
    ataques = [s for s, r in zip(teste_scores, teste_rotulos) if r == 0]
    pares_certos = 0
    for sr in reais:
        vence = sum(sr > sa for sa in ataques) + 0.5 * sum(sr == sa for sa in ataques)
        print(f"  real {sr:.2f} tem score maior que {vence:g} de {len(ataques)} ataques")
        pares_certos += vence
    total_pares = len(reais) * len(ataques)
    print(f"AUC (à mão)     = {pares_certos:g}/{total_pares} = {pares_certos / total_pares:.4f}")
    print(f"AUC (sklearn)   = {calcular_auc(teste_scores, teste_rotulos):.4f}\n")


if __name__ == "__main__":
    limiar = mostrar_tabela_dev()
    mostrar_decisoes_teste(limiar)
    mostrar_contas(limiar)
    mostrar_eer_auc_teste()

    print("PASSO 5 - tudo de uma vez com avaliar_dev_teste()")
    resultado = avaliar_dev_teste(dev_scores, dev_rotulos,
                                  teste_scores, teste_rotulos, teste_tipos)
    for chave, valor in resultado.items():
        print(f"  {chave:13s} {valor:.4f}")
