import os
import sys
import pandas as pd

from utils import ler_config, calcular_apcer_bpcer, PASTA_PROJETO
from resultados import num, salvar_tabela, mostrar, ESTRATEGIAS

sys.path.insert(0, os.path.join(PASTA_PROJETO, "pi"))
from estrategias_pi import previsto, CONJUNTO

COMPONENTES = {"deteccao": "Detecção de face (dlib HOG)", "recorte": "Recorte + normalização",
               "rede": "Rede (MobileNetV3-Small)", "landmarks": "68 landmarks (dlib)", "ear_regra": "EAR + regra",
               "ramo_a": "Ramo A (detecção + recorte + rede)", "ramo_b": "Ramo B (detecção + landmarks + EAR)",
               "a_mais_b": "A+B (todos os componentes)"}
NECESSARIO = {"ramo_a": "10", "ramo_b": "30", "a_mais_b": "30"}


def milhar(n):
    return f"{int(n):,}".replace(",", ".")


def tabela_modelo(info, config):
    caminho = os.path.join(PASTA_PROJETO, config["preditor_landmarks"])
    return pd.DataFrame([
        {"Modelo": "MobileNetV3-Small + cabeça (ramo A)", "Arquivo (MB)": num(info["arquivo_mb"]),
         "Parâmetros": f"{milhar(info['parametros'])} ({milhar(info['parametros_cabeca'])} na cabeça)",
         "FLOPs por recorte": f"{num(info['flops'] / 1e6, 1)} M", "MACs por recorte": f"{num(info['macs'] / 1e6, 1)} M"},
        {"Modelo": "Preditor de 68 landmarks do dlib (ramo B)", "Arquivo (MB)": num(os.path.getsize(caminho) / 1024 ** 2),
         "Parâmetros": "--", "FLOPs por recorte": "--", "MACs por recorte": "--"},
        {"Modelo": "Detector HOG do dlib (ramos A e B)", "Arquivo (MB)": "embutido no dlib",
         "Parâmetros": "--", "FLOPs por recorte": "--", "MACs por recorte": "--"}])


def tabela_latencia(resumo):
    return pd.DataFrame([{"Etapa": nome, "Mediana (ms)": num(resumo.loc[c, "mediana_ms"]),
                          "P90 (ms)": num(resumo.loc[c, "p90_ms"]), "Média (ms)": num(resumo.loc[c, "media_ms"]),
                          "Quadros/s": num(resumo.loc[c, "quadros_por_segundo"], 1),
                          "Quadros/s necessários": NECESSARIO.get(c, "--")} for c, nome in COMPONENTES.items()])


def tabela_estrategias(t):
    linhas = []
    for estrategia in ESTRATEGIAS:
        g = t[t["estrategia"] == estrategia]
        linhas.append({"Estratégia": estrategia, "Consulta B (%)": num(100 * g["consultou_b"].mean()),
                       "Cobertura A (%)": num(100 * (1 - g["consultou_b"].mean())),
                       "Observação (s)": num(g["observacao_s"].mean()),
                       "CPU por vídeo (s)": num(g["cpu_s"].mean()), "CPU previsto (s)": num(g["cpu_prev"].mean()),
                       "Até a decisão (s)": num(g["decisao_s"].mean()),
                       "Espera média (s)": num(g["resposta_s"].mean()),
                       "Espera p90 (s)": num(g["resposta_s"].quantile(0.9)),
                       "Espera máx (s)": num(g["resposta_s"].max()),
                       "Vídeos em tempo real (%)": num(100 * (g["cpu_s"] <= g["observacao_s"]).mean(), 1)})
    return pd.DataFrame(linhas)


def tabela_deteccao(t):
    linhas = []
    for estrategia in ESTRATEGIAS:
        g = t[t["estrategia"] == estrategia]
        pi = calcular_apcer_bpcer(g["decisao"], g["rotulo"], g["tipo"], 0.5)
        pc = calcular_apcer_bpcer(g["decisao_pc"], g["rotulo"], g["tipo"], 0.5)
        linhas.append({"Estratégia": estrategia, "APCER print (%)": num(100 * pi["apcer_print"]),
                       "APCER replay (%)": num(100 * pi["apcer_replay"]), "APCER máx (%)": num(100 * pi["apcer_max"]),
                       "BPCER (%)": num(100 * pi["bpcer"]), "ACER Pi (%)": num(100 * pi["acer"]),
                       "ACER PC (%)": num(100 * pc["acer"]),
                       "Decisões iguais ao PC": f"{(g['decisao'] == g['decisao_pc']).sum()}/{len(g)}"})
    return pd.DataFrame(linhas)


def tabela_correcoes(t):
    a = t[t["estrategia"] == "só A"].set_index("video")["decisao"]
    g = t[(t["estrategia"] == "guiada") & (t["consultou_b"] == 1)]
    a_certo = a.loc[g["video"]].values == g["rotulo"].values
    g_certo = (g["decisao"] == g["rotulo"]).values
    return pd.DataFrame([{"Vídeos": t["video"].nunique(), "Consultados": len(g),
                          "A errava": int((~a_certo).sum()), "Corrigidos": int((~a_certo & g_certo).sum()),
                          "Estragados": int((a_certo & ~g_certo).sum()),
                          "Consultados que eram ataque": int((g["rotulo"] == 0).sum())}])


def tabela_memoria(m):
    nomes = [("inicio", "Python + PyTorch carregados"), ("com_mobilenet", "+ MobileNetV3 (ramo A)"),
             ("com_dlib", "+ detector e landmarks do dlib"), ("processando_media", "Processando quadro a quadro (média)"),
             ("processando_pico", "Processando quadro a quadro (pico)"),
             ("buffer_video_mb", f"Buffer de um vídeo inteiro ({int(m['buffer_quadros'])} quadros, guiada)"),
             ("pico_com_buffer", "Pico com o buffer")]
    return pd.DataFrame([{"Momento": nome, "Memória (MB)": ("+" if c == "buffer_video_mb" else "") + num(m[c], 0)}
                         for c, nome in nomes])


def tabela_condicoes(q, t):
    return pd.DataFrame([
        {"Medição": "Latência por componente (etapa 13)", "Vídeos": q["video"].nunique(),
         "Repetições": q["repeticao"].nunique(), "Quadros medidos": milhar(len(q)),
         "Temperatura (°C)": f"{num(q['temperatura_c'].min(), 1)} a {num(q['temperatura_c'].max(), 1)}",
         "Clock (MHz)": (f"{q['clock_mhz'].min():.0f} (fixo)" if q["clock_mhz"].nunique() == 1
                         else f"{q['clock_mhz'].min():.0f} a {q['clock_mhz'].max():.0f}")},
        {"Medição": "Estratégias a 30 fps (etapa 14)", "Vídeos": t["video"].nunique(), "Repetições": 1,
         "Quadros medidos": milhar(t.groupby("video")["quadros_observados"].first().sum()),
         "Temperatura (°C)": f"{num(t['temperatura_c'].min(), 1)} a {num(t['temperatura_c'].max(), 1)}",
         "Clock (MHz)": "--"}])


if __name__ == "__main__":
    config = ler_config()
    pasta_pi = os.path.join(PASTA_PROJETO, config["resultados_dir"], "pi")
    saida = os.path.join(PASTA_PROJETO, config["resultados_dir"], "final")
    os.makedirs(saida, exist_ok=True)
    info = pd.read_csv(os.path.join(pasta_pi, "modelo_info.csv")).iloc[0]
    resumo = pd.read_csv(os.path.join(pasta_pi, "medicoes_resumo.csv"), index_col="medida")
    memoria = pd.read_csv(os.path.join(pasta_pi, "medicoes_memoria.csv")).iloc[0]
    quadros = pd.read_csv(os.path.join(pasta_pi, "medicoes_quadros.csv"), dtype={"video": str})
    t = pd.read_csv(os.path.join(pasta_pi, f"estrategias_pi_{CONJUNTO}_videos.csv"), dtype={"video": str})

    custos = resumo["media_ms"] / 1000
    prev = [previsto(n, e, c, custos, config["passo_quadros"])
            for n, e, c in zip(t["quadros_observados"], t["estrategia"], t["consultou_b"])]
    t["cpu_prev"] = [cpu for _, cpu in prev]

    fold = CONJUNTO.replace("p", "P").replace("_f", ", fold ")
    tabelas = [
        (tabela_modelo(info, config), "tabela_pi_modelo", "Tamanho e complexidade dos modelos embarcados", {}),
        (tabela_latencia(resumo), "tabela_pi_latencia", "Latência por quadro no Raspberry Pi 5 "
         f"({fold}, {quadros['video'].nunique()} vídeos, {quadros['repeticao'].nunique()} repetições)", {}),
        (tabela_estrategias(t), "tabela_pi_estrategias", f"Custo e tempo de resposta das estratégias no Raspberry Pi 5 "
         f"com chegada dos quadros a 30 fps ({fold}, teste completo); CPU previsto pela latência por componente",
         {"ajustar": True}),
        (tabela_deteccao(t), "tabela_pi_deteccao", f"Detecção no Raspberry Pi 5 e no PC ({fold}, teste completo)",
         {"ajustar": True}),
        (tabela_correcoes(t), "tabela_pi_correcoes", f"Vídeos em que a guiada consultou o ramo B no Raspberry Pi 5 "
         f"({fold})", {"texto": 0}),
        (tabela_memoria(memoria), "tabela_pi_memoria", "Memória do processo (RSS) no Raspberry Pi 5", {}),
        (tabela_condicoes(quadros, t), "tabela_pi_condicoes", "Condições das medições no Raspberry Pi 5", {}),
    ]
    for tabela, nome, legenda, opcoes in tabelas:
        salvar_tabela(tabela, saida, nome, legenda, **opcoes)
        mostrar(f"{nome}: {legenda}", tabela)
    print("\nArquivos salvos em", saida)
