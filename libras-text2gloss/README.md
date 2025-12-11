# PT-BR to LIBRAS Text2Gloss Translation System

Sistema avançado de tradução de Português Brasileiro para glossas LIBRAS (Língua Brasileira de Sinais) usando **mT5**, **transfer learning multilíngue** e **embeddings syntax-aware**.

## 🎯 Objetivo

Criar um sistema de tradução texto→gloss que supera o estado da arte através de:

- **Modelos Maiores**: mT5-base/large ao invés de Transformers menores
- **Treinamento Multilíngue**: Aproveitar datasets ASL, DGS e LIBRAS
- **Transfer Learning**: Pré-treinar em múltiplas línguas de sinais
- **Syntax-Aware**: Injetar informação sintática do Português

## 📊 Datasets

### LIBRAS (Principal)
- **VLibrasBD**: 127,349 pares PT-BR ↔ LIBRAS gloss
- **libras_bilingual_dataset**: 68,029 pares alinhados
- Fonte: Governo Federal Brasileiro, conteúdo geral

### Multilíngue (Transfer Learning)
- **RWTH-PHOENIX-2014-T**: Alemão → DGS (7,096 treino)
- **ASL Datasets**: Inglês → ASL (opcional)

## 🏗️ Arquitetura

```
Input: "O menino está jogando bola"
  ↓
[Tokenização mT5] + [Extração Sintática spaCy]
  ↓
[Word Embeddings] + [Syntax Embeddings]
  ↓
[L2 Normalization & Sum]
  ↓
[mT5 Encoder-Decoder]
  ↓
Output: "MENINO JOGAR BOLA"
```

## 🚀 Instalação

### 1. Criar ambiente virtual

```bash
cd libras-text2gloss
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Baixar modelo spaCy para Português

```bash
python -m spacy download pt_core_news_lg
```

### 4. Configurar WandB (opcional)

```bash
wandb login
```

## 📁 Estrutura do Projeto

```
libras-text2gloss/
├── configs/              # Arquivos de configuração YAML
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
├── data/                 # Datasets
│   ├── raw/             # Dados brutos
│   ├── processed/       # Dados processados
│   └── multilingual/    # Datasets combinados
├── models/              # Modelos e checkpoints
│   ├── checkpoints/     # Checkpoints de treinamento
│   └── pretrained/      # Modelos pré-treinados
├── utils/               # Utilitários
│   ├── data_loader.py
│   ├── syntax_processor.py
│   ├── tokenizer_utils.py
│   └── metrics.py
├── scripts/             # Scripts de treinamento/avaliação
│   ├── train_mt5_syntax.py
│   ├── train_mt5_baseline.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/           # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── model_analysis.ipynb
└── tests/               # Testes unitários
```

## 🎓 Uso

### Preparar Dados

```bash
python scripts/preprocess_data.py \
  --input data/raw/vlibrasbd.csv \
  --output data/processed/ \
  --split 0.8 0.1 0.1
```

### Treinar Modelo Baseline

```bash
python scripts/train_mt5_baseline.py \
  --config configs/training_config.yaml \
  --output models/checkpoints/baseline
```

### Treinar Modelo Syntax-Aware

```bash
python scripts/train_mt5_syntax.py \
  --config configs/training_config.yaml \
  --output models/checkpoints/syntax_aware
```

### Avaliar Modelos

```bash
python scripts/evaluate.py \
  --model_baseline models/checkpoints/baseline/best_model \
  --model_syntax models/checkpoints/syntax_aware/best_model \
  --test_data data/processed/test.json
```

### Inferência Interativa

```bash
python scripts/inference.py \
  --model models/checkpoints/syntax_aware/best_model \
  --interactive
```

## 📈 Métricas de Avaliação

- **sacreBLEU**: Qualidade de tradução
- **TER**: Taxa de erro de tradução
- **ROUGE-L F1**: Sobreposição de sequências
- **METEOR**: Correlação com julgamento humano
- **Generation Time**: Tempo de geração

## 🔬 Experimentos

### Baseline vs Syntax-Aware

Comparação entre modelo padrão mT5 e versão com informação sintática.

### Transfer Learning

1. **Fase 1**: Pré-treinar em ASL+DGS
2. **Fase 2**: Fine-tune em LIBRAS
3. **Fase 3**: Adaptação de domínio

### Ablation Studies

- Impacto de diferentes tags sintáticas
- Tamanho do modelo (base vs large)
- Quantidade de dados multilíngues

## 📚 Referências

- **Artigo Base**: Egea Gómez et al. (2021) - Syntax-aware Transformers for Text to Sign Gloss Translation
- **mT5**: Xue et al. (2021) - mT5: A massively multilingual pre-trained text-to-text transformer
- **VLibrasBD**: Corpus bilíngue PT-BR ↔ LIBRAS

## 📝 Licença

MIT License

## 👥 Contribuidores

Baseado no trabalho original de Syntax-Aware Transformer Text2Gloss (LaSTUS-TALN-UPF).

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.
