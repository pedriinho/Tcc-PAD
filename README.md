# TCC

Código do meu TCC na UFAL sobre detecção de ataques de apresentação facial no Raspberry Pi 5. Comparo um ramo espacial (MobileNetV3-Small), um ramo temporal (piscada) e as estratégias só A, só B, A+B sempre e guiada por confiança.

## O que precisa antes de começar

O repositório só tem o código. Faltam três coisas, que não coloquei aqui:

1. **O dataset OULU-NPU.** A licença não permite redistribuir, então é preciso ter acesso próprio a ele. A pasta deve conter `Train_files`, `Dev_files`, `Test_files` e `Protocols`.
2. **O modelo de landmarks do dlib.** Baixe o `shape_predictor_68_face_landmarks.dat.bz2` em dlib.net/files, descompacte e coloque em `modelos/`.
3. **Os caminhos no `config.yaml`.** Troque `dataset_dir` e `preproc_dir`, que ainda apontam para as pastas do meu computador.

## Instalação no PC

Usei Windows 11, Python 3.10 e uma RTX 3060 Ti.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas opencv-python scikit-learn matplotlib pyyaml cmake dlib
```

Instale o PyTorch primeiro e com esse `--index-url`; sem ele, vem a versão só para CPU. O dlib é compilado na instalação, então no Windows também precisa do Visual Studio Build Tools com "Desenvolvimento para desktop com C++". As versões exatas que usei estão no `requirements.txt`.

Para conferir se ficou tudo certo, a pasta `verificações` tem dois scripts que ajudam:

```powershell
python verificações/checar_ambiente.py
python verificações/conferir_protocolos.py
```

## Rodando no PC

A ordem importa, porque cada script usa o que o anterior gerou:

```powershell
python preprocessamento.py
python treino_espacial.py
python avalia_espacial.py
python avalia_temporal.py
python estrategias.py
python resultados.py
```

O que cada um faz:

- `preprocessamento.py` detecta a face em todos os quadros dos 4.950 vídeos e salva os recortes e os landmarks. Aqui levou umas 2 horas com 6 processos. Se parar no meio, continua de onde estava.
- `treino_espacial.py` treina os 14 modelos (P1, P2 e os 6 folds de P3 e de P4), com 10 épocas cada e a taxa de aprendizado caindo até zero. Fica guardada a época de menor EER no conjunto de desenvolvimento. Levou cerca de 3 horas, e modelos já terminados são pulados.
- `avalia_espacial.py` e `avalia_temporal.py` calculam os scores e as métricas de cada ramo.
- `estrategias.py` compara as quatro estratégias.
- `resultados.py` gera as tabelas e os gráficos do texto.

Os limiares são sempre escolhidos no desenvolvimento e depois aplicados no teste, sem ajuste.

## Rodando no Raspberry Pi

Usei um Pi 5 de 8 GB, com ventoinha e sem acelerador, no Raspberry Pi OS 64 bits e Python 3.13. Lá instalei o PyTorch só para CPU, `opencv-python-headless`, `dlib`, `pandas`, `scikit-learn` e `pyyaml`.

As medições usam o modelo do P3 fold 5. Com o pipeline do PC já rodado, copie para o Pi, mantendo as mesmas pastas:

- o código e o `config.yaml`;
- `modelos/espacial_p3_f5.pt` e o modelo de landmarks;
- `resultados/espacial/resumo.csv` e os CSVs de `resultados/espacial/p3_f5/`;
- `resultados/temporal/resumo.csv` e `resultados/temporal/p3_f5/scores_videos.csv`;
- em `pi/videos/`, os 300 vídeos de `Test_files` que começam com `5_` (cerca de 6,5 GB);
- em `pi/amostras/<vídeo>/`, os recortes dos 18 vídeos listados em `VIDEOS`, no `pi/medicoes_pi.py`. Eles vêm de `preproc_dir/recortes/<vídeo>/`, e o script só usa os de 1 a cada 3 quadros.

Depois, no Pi:

```bash
python pi/modelo_pi.py
python pi/medicoes_pi.py
python pi/estrategias_pi.py
```

- `modelo_pi.py` mede tamanho, parâmetros e FLOPs, e confere se o Pi dá os mesmos scores que o PC.
- `medicoes_pi.py` mede a latência de cada etapa e a memória, em cerca de 10 minutos.
- `estrategias_pi.py` roda as quatro estratégias nos 300 vídeos, simulando a chegada dos quadros a 30 fps. Leva umas 2 horas. Ele grava vídeo a vídeo, então dá para interromper e continuar. Para começar do zero, apague `resultados/pi/estrategias_pi_p3_f5_videos.csv`.

Quando terminar, traga `resultados/pi/` de volta para o PC e rode:

```powershell
python resultados.py
python resultados_pi.py
```

## De onde vem o orçamento da guiada

O `orcamento_consulta: 0.45` do `config.yaml` não foi escolhido olhando o resultado. Ele sai dos tempos medidos no Pi para um vídeo de 151 quadros: o ramo A custa 3,07 s, a consulta ao B acrescenta 4,30 s e o vídeo dura 5,03 s na taxa nominal. Para o custo médio caber nesse tempo, a fração consultada não pode passar de 45,6%, e usei 45%. Esse valor é calibrado no desenvolvimento, e no teste a fração consultada acabou maior; isso está discutido no texto do TCC.

## Outros scripts

A pasta `verificações` não faz parte do pipeline. São scripts que usei para conferir coisas ou justificar escolhas:

- `checar_ambiente.py`: versões das bibliotecas e teste da GPU.
- `conferir_protocolos.py`: conta os vídeos dos protocolos e verifica se os arquivos existem.
- `teste_metricas.py`: exemplo pequeno, com 8 vídeos, para conferir as métricas à mão.
- `teste_face.py`: roda a detecção em um vídeo e salva imagens anotadas, por exemplo `python verificações/teste_face.py 1_1_01_1`.
- `analise_olhos.py`: analisa o EAR no treino e gera a tabela usada para escolher o fator 0,75 do olho fechado.
- `teste_decaimento.py`: teste no P1 que comparei com a taxa de aprendizado fixa antes de adotar o decaimento.

## Onde ficam as saídas

Tudo vai para `resultados/`: `treino/`, `espacial/`, `temporal/`, `estrategias/`, `pi/` e `final/`. As tabelas e os gráficos usados no texto estão em `final/`. Os modelos treinados ficam em `modelos/`. Essas pastas estão no `.gitignore`, assim como `pi/videos/` e `pi/amostras/`, que têm conteúdo do dataset e não podem ser publicadas.