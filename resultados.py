import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from utils import ler_config, PASTA_PROJETO
from treino_espacial import CONJUNTOS

FAIXAS_CONFIANCA = [0, 1, 2, 4, 8, np.inf]
ESTRATEGIAS = ["só A", "só B", "A+B sempre", "guiada"]
CORES = {"A": "#2a78d6", "B": "#eb6834", "A+B": "#1baf7a"}
DETECCAO = [("apcer_print", "APCER print (%)"), ("apcer_replay", "APCER replay (%)"),
            ("apcer_max", "APCER máx (%)"), ("bpcer", "BPCER (%)"), ("acer", "ACER (%)")]
COMPLEMENTO = [("eer_dev", "EER dev (%)"), ("eer_teste", "EER teste (%)"), ("auc_teste", "AUC"),
               ("consulta_b", "Consulta B (%)"), ("cobertura_a", "Cobertura A (%)")]
PONTOS = [("consulta_b", "Consulta B (%)"), ("apcer_max", "APCER máx (%)"), ("bpcer", "BPCER (%)"), ("acer", "ACER (%)")]


def num(x, casas=2):
    return "--" if pd.isna(x) else f"{x:.{casas}f}".replace(".", ",")


def media_desvio(valores, coluna=""):
    escala, casas = (1, 3) if coluna == "auc_teste" else (100, 2)
    v = escala * pd.Series(valores, dtype=float).dropna()
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return num(v.iloc[0], casas)
    return f"{num(v.mean(), casas)} $\\pm$ {num(v.std(ddof=0), casas)}"


def salvar_tabela(df, pasta, nome, legenda, texto=1, grupo=None, ajustar=False):
    df.to_csv(os.path.join(pasta, nome + ".csv"), index=False)
    esc = lambda s: str(s).replace("%", r"\%")
    linhas = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{legenda}}}",
              f"\\label{{tab:{nome.replace('tabela_', '')}}}"] + ([r"\resizebox{\textwidth}{!}{%"] if ajustar else [])
    linhas += [r"\begin{tabular}{" + "l" * texto + "r" * (len(df.columns) - texto) + "}", r"\toprule",
               " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    for i, (_, linha) in enumerate(df.iterrows()):
        if grupo and i > 0 and linha[grupo] != df[grupo].iloc[i - 1]:
            linhas.append(r"\midrule")
        linhas.append(" & ".join(esc(v) for v in linha) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}" + ("}" if ajustar else ""), r"\end{table}"]
    with open(os.path.join(pasta, nome + ".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


def ler_por_protocolo(pasta, arquivo):
    tabelas = {}
    for protocolo, fold in CONJUNTOS:
        nome = f"p{protocolo}" + (f"_f{fold}" if fold else "")
        t = pd.read_csv(os.path.join(pasta, "estrategias", nome, arquivo), dtype={"video": str})
        tabelas.setdefault(protocolo, []).append(t.assign(fold=fold))
    return {p: pd.concat(lista) for p, lista in tabelas.items()}


def tabela_por_protocolo(dados, coluna_nome, titulo_nome, nomes, metricas):
    linhas = []
    for protocolo, grupo in dados.groupby("protocolo"):
        for nome in nomes:
            g = grupo[grupo[coluna_nome] == nome]
            linhas.append({"Protocolo": f"P{protocolo}", titulo_nome: nome,
                           **{titulo: media_desvio(g[c], c) for c, titulo in metricas}})
    return pd.DataFrame(linhas)


def tabela_correcoes(corr, decisoes):
    linhas = []
    for protocolo, c in corr.groupby("protocolo"):
        t = decisoes[protocolo][decisoes[protocolo]["split"] == "teste"]
        s = c[["consultados", "a_errava", "corrigidos", "estragados"]].sum()
        linhas.append({"Protocolo": f"P{protocolo}", "Vídeos de teste": len(t),
                       "Erros do A": int((t["dec_a"] != t["rotulo"]).sum()), "Consultados": s["consultados"],
                       "Consultados (%)": num(100 * s["consultados"] / len(t)), "A errava": s["a_errava"],
                       "Corrigidos": s["corrigidos"], "Estragados": s["estragados"],
                       "Saldo": s["corrigidos"] - s["estragados"]})
    return pd.DataFrame(linhas)


def nome_faixa(faixa):
    return f"$\\geq$ {faixa.left:g}" if np.isinf(faixa.right) else f"[{faixa.left:g}; {faixa.right:g})"


def erro_por_faixa(decisoes):
    linhas = []
    for protocolo, d in decisoes.items():
        t = d[d["split"] == "teste"]
        t = t.assign(faixa=pd.cut(t["conf_a"], FAIXAS_CONFIANCA, right=False))
        for faixa, g in t.groupby("faixa", observed=True):
            linhas.append({"protocolo": protocolo, "faixa": nome_faixa(faixa), "videos": len(g),
                           "fracao_videos": len(g) / len(t), "reais": int(g["rotulo"].sum()),
                           "erro_A": (g["dec_a"] != g["rotulo"]).mean(),
                           "erro_B": (g["dec_b"] != g["rotulo"]).mean(),
                           "erro_A+B": (g["dec_ab"] != g["rotulo"]).mean()})
    return pd.DataFrame(linhas)


def tabela_faixas(faixas):
    return pd.DataFrame({"Protocolo": "P" + faixas["protocolo"].astype(str), "Confiança do A": faixas["faixa"],
                         "Vídeos": faixas["videos"], "Vídeos (%)": (100 * faixas["fracao_videos"]).map(num),
                         "Reais": faixas["reais"], "Erro A (%)": (100 * faixas["erro_A"]).map(num),
                         "Erro B (%)": (100 * faixas["erro_B"]).map(num),
                         "Erro A+B (%)": (100 * faixas["erro_A+B"]).map(num)})


def grafico_faixas(faixas, caminho):
    fig, eixos = plt.subplots(1, 4, figsize=(20, 4.2), sharey=True)
    for eixo, (protocolo, f) in zip(eixos, faixas.groupby("protocolo")):
        x = np.arange(len(f))
        for i, ramo in enumerate(["A", "B", "A+B"]):
            eixo.bar(x + (i - 1) * 0.27, f[f"erro_{ramo}"], width=0.25, color=CORES[ramo],
                     label="A+B sempre" if ramo == "A+B" else ramo)
        eixo.set_xticks(x, [f"{fx}\nn={n}" for fx, n in zip(f["faixa"], f["videos"])], fontsize=8)
        eixo.set_title(f"P{protocolo}")
        eixo.set_xlabel("confiança do ramo A (logit)")
        eixo.grid(alpha=0.3, axis="y")
    eixos[0].set_ylabel("erro (fração de vídeos de teste)")
    eixos[0].legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def grafico_roc(decisoes, caminho):
    fig, eixos = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    for eixo, (protocolo, d) in zip(eixos, decisoes.items()):
        teste = d[d["split"] == "teste"]
        for coluna, nome, cor in [("score_a", "só A", CORES["A"]), ("score_b", "só B", CORES["B"]),
                                  ("score_ab", "A+B sempre", CORES["A+B"])]:
            for i, (_, t) in enumerate(teste.groupby("fold", dropna=False)):
                fpr, tpr, _ = roc_curve(t["rotulo"], t[coluna])
                eixo.plot(100 * fpr, 100 * tpr, color=cor, linewidth=1.2, alpha=0.6, label=nome if i == 0 else None)
        eixo.plot([0, 100], [0, 100], color="#999999", linestyle=":", linewidth=1)
        eixo.set_title(f"P{protocolo}")
        eixo.set_xlabel("ataques aceitos como reais (%)")
        eixo.grid(alpha=0.3)
    eixos[0].set_ylabel("reais aceitos (100 − BPCER, %)")
    eixos[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def custos_pi(pasta):
    caminho = os.path.join(pasta, "pi", "medicoes_resumo.csv")
    if not os.path.isfile(caminho):
        return None
    r = pd.read_csv(caminho, index_col="medida")["media_ms"] / 1000
    n, n_a = 151, 51
    return n_a * r["ramo_a"], n * (r["landmarks"] + r["ear_regra"]) + (n - n_a) * r["deteccao"], n / 30


def grafico_curva(curvas, pontos, custos, caminho):
    x = (lambda f: custos[0] + f * custos[1]) if custos else (lambda f: 100 * f)
    fig, eixos = plt.subplots(2, 2, figsize=(11, 8))
    for eixo, (protocolo, c) in zip(eixos.flat, curvas.items()):
        m = c.groupby("orcamento")[["consulta_teste", "acer_teste"]].mean()
        eixo.plot(x(m["consulta_teste"]), 100 * m["acer_teste"], color="#6b6b6b", marker="o", markersize=4,
                  linewidth=1.5, label="guiada (orçamentos de 0 até o escolhido)")
        g = pontos[pontos["protocolo"] == protocolo].groupby("ponto", sort=False)[["consulta_b", "acer"]].mean()
        for nome, cor in zip(g.index, [CORES["A"], CORES["B"], "#000000", CORES["A+B"]]):
            eixo.scatter(x(g.loc[nome, "consulta_b"]), 100 * g.loc[nome, "acer"], color=cor, s=70, zorder=3,
                         label=nome)
        if custos:
            eixo.axvline(custos[2], color="#999999", linestyle=":", linewidth=1, label="tempo disponível (observação)")
        eixo.set_title(f"P{protocolo}")
        eixo.set_xlabel("custo médio de CPU por vídeo no Raspberry Pi 5 (s)" if custos
                        else "vídeos de teste que consultaram o ramo B (%)")
        eixo.set_ylabel("ACER no teste (%)")
        eixo.grid(alpha=0.3)
    eixos[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def mostrar(titulo, tabela):
    print(f"\n{titulo}")
    print(tabela.to_string(index=False).replace("$\\pm$", "±").replace("$\\geq$", ">="))


if __name__ == "__main__":
    config = ler_config()
    pasta = os.path.join(PASTA_PROJETO, config["resultados_dir"])
    saida = os.path.join(pasta, "final")
    os.makedirs(saida, exist_ok=True)

    resumo = pd.read_csv(os.path.join(pasta, "estrategias", "resumo.csv"))
    resumo["cobertura_a"] = 1 - resumo["consulta_b"]
    colunas = [c for c, _ in DETECCAO + COMPLEMENTO]
    resumo.groupby(["protocolo", "estrategia"], sort=False)[colunas].mean().to_csv(
        os.path.join(saida, "media_por_protocolo.csv"))
    t = tabela_por_protocolo(resumo, "estrategia", "Estratégia", ESTRATEGIAS, DETECCAO)
    salvar_tabela(t, saida, "tabela_estrategias", "Detecção no teste por protocolo e estratégia "
                  "(P3 e P4: média $\\pm$ desvio entre os 6 folds)", texto=2, grupo="Protocolo", ajustar=True)
    mostrar("Detecção no teste (%):", t)
    t = tabela_por_protocolo(resumo, "estrategia", "Estratégia", ESTRATEGIAS, COMPLEMENTO)
    salvar_tabela(t, saida, "tabela_estrategias_complementar", "EER, AUC e uso do ramo B por protocolo e "
                  "estratégia (a guiada produz decisões, não scores, e por isso não tem EER nem AUC)",
                  texto=2, grupo="Protocolo", ajustar=True)
    mostrar("EER, AUC e consulta ao B:", t)

    pontos = pd.read_csv(os.path.join(pasta, "estrategias", "pontos.csv"))
    t = tabela_por_protocolo(pontos, "ponto", "Ponto", list(dict.fromkeys(pontos["ponto"])), PONTOS)
    salvar_tabela(t, saida, "tabela_pontos", "Estratégia guiada: ponto de operação pelo orçamento de custo e "
                  "referências (só A, menor erro no dev e A+B sempre), no teste", texto=2, grupo="Protocolo")
    mostrar("Guiada: pontos de operação e referências (%):", t)
    grafico_curva(ler_por_protocolo(pasta, "curva_orcamento.csv"), pontos, custos_pi(pasta),
                  os.path.join(saida, "acer_x_custo.png"))

    decisoes = ler_por_protocolo(pasta, "decisoes_videos.csv")
    corr = pd.read_csv(os.path.join(pasta, "estrategias", "correcoes.csv"))
    t = tabela_correcoes(corr, decisoes)
    salvar_tabela(t, saida, "tabela_correcoes", "Vídeos de teste em que a guiada consultou o ramo B "
                  "(soma dos folds): erros do ramo A corrigidos e acertos estragados pela fusão A+B")
    mostrar("Correções do ramo B no ponto de operação (soma dos folds):", t)

    faixas = erro_por_faixa(decisoes)
    faixas.to_csv(os.path.join(saida, "erro_por_faixa.csv"), index=False)
    grafico_faixas(faixas, os.path.join(saida, "erro_por_faixa_confianca.png"))
    t = tabela_faixas(faixas)
    salvar_tabela(t, saida, "tabela_erro_faixa", "Erro no teste por faixa de confiança do ramo A "
                  "(|logit(score) $-$ logit(limiar)|; todos os folds juntos)", texto=2, grupo="Protocolo")
    mostrar("Erro por faixa de confiança do ramo A:", t)

    grafico_roc(decisoes, os.path.join(saida, "roc_teste.png"))
    print("\nArquivos salvos em", saida)
