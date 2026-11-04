import os
import sys
import platform
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BIBLIOTECAS = ["numpy", "pandas", "cv2", "dlib", "torch", "torchvision",
               "sklearn", "matplotlib", "yaml"]

PASTAS_OULU = ["Train_files", "Dev_files", "Test_files", "Protocols"]


def mostrar_versoes():
    print("=== Versões ===")
    print("Python:", sys.version.split()[0])
    print("Sistema:", platform.system(), platform.release())
    faltando = []
    for nome in BIBLIOTECAS:
        try:
            modulo = importlib.import_module(nome)
            versao = getattr(modulo, "__version__", "?")
            print(f"  {nome:12s} {versao}")
        except ImportError:
            print(f"  {nome:12s} NÃO INSTALADO")
            faltando.append(nome)
    return faltando


def checar_gpu():
    import torch
    print("\n=== GPU ===")
    print("CUDA disponível:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("  Atenção: o PyTorch instalado não enxerga a GPU (versão só CPU?)")
        return
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA do PyTorch:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    memoria = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Memória da GPU: {memoria:.1f} GB")
    x = torch.rand(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("Teste de cálculo na GPU: ok", tuple(y.shape))


def checar_config():
    from utils import ler_config, PASTA_PROJETO
    config = ler_config()
    print("\n=== config.yaml ===")
    for chave, valor in config.items():
        print(f"  {chave}: {valor}")

    dataset_dir = config["dataset_dir"]
    print("\nPasta do OULU-NPU existe:", os.path.isdir(dataset_dir))
    for pasta in PASTAS_OULU:
        caminho = os.path.join(dataset_dir, pasta)
        print(f"  {pasta:12s} {'ok' if os.path.isdir(caminho) else 'NÃO ENCONTRADA'}")

    resultados_dir = os.path.join(PASTA_PROJETO, config["resultados_dir"])
    os.makedirs(resultados_dir, exist_ok=True)
    print("Pasta de resultados:", resultados_dir)


if __name__ == "__main__":
    faltando = mostrar_versoes()
    if "torch" in faltando:
        print("\nPyTorch não instalado: pulando o teste da GPU")
    else:
        checar_gpu()
    if "yaml" in faltando:
        print("\nPyYAML não instalado: pulando a leitura do config.yaml")
    else:
        checar_config()
    print("\nFaltando:", faltando if faltando else "nada, ambiente pronto!")
