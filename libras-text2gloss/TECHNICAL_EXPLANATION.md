# Sistema de Tradução PT-BR para LIBRAS: Explicação Técnica

## Visão Geral do Sistema

Este documento explica o sistema implementado de tradução de Português Brasileiro para glossas LIBRAS, comparando com o trabalho original e destacando as melhorias.

---

## 1. O PROBLEMA

### Trabalho Original (Alemão → DGS)
- **Língua Fonte**: Alemão
- **Língua Alvo**: DGS (German Sign Language) glosses
- **Dataset**: RWTH-PHOENIX-2014-T (7,096 amostras de treino)
- **Domínio**: Previsão do tempo
- **Desafio**: Traduzir texto falado para representação gloss de língua de sinais

### Nossa Proposta (Português BR → LIBRAS)
- **Língua Fonte**: Português Brasileiro
- **Língua Alvo**: LIBRAS (Brazilian Sign Language) glosses
- **Dataset**: VLibrasBD (127,349 pares PT-BR ↔ LIBRAS)
- **Domínio**: Geral + conteúdo governamental
- **Desafio**: Mesmo problema, mas com dataset maior e mais diverso

---

## 2. A SOLUÇÃO PROPOSTA

### Arquitetura Original
```
Texto Alemão
    ↓
[Tokenização SentencePiece]
    ↓
[Word Embeddings] + [Dependency Embeddings (spaCy alemão)]
    ↓
[L2 Normalization & Sum]
    ↓
[mBART Encoder-Decoder]
    ↓
Glossas DGS
```

**Tecnologias**: TensorFlow 2.4, mBART, spaCy alemão

### Nossa Arquitetura (Melhorada)
```
Texto Português BR
    ↓
[Tokenização mT5]
    ↓
[Word Embeddings] + [Dependency Embeddings (spaCy português)]
    ↓
[L2 Normalization & Sum]
    ↓
[mT5 Encoder-Decoder]
    ↓
Glossas LIBRAS
```

**Tecnologias**: PyTorch 2.0+, mT5, spaCy português

---

## 3. MELHORIAS IMPLEMENTADAS

### 3.1. Modelo Maior e Multilíngue (mT5)

**Original**: mBART (modelo multilíngue BART)
- Focado em tradução
- Pré-treinado em 25 línguas

**Nossa Implementação**: mT5 (Multilingual T5)
- Arquitetura text-to-text unificada
- Pré-treinado em 101 línguas
- Melhor para transfer learning
- Tamanhos disponíveis: small (300M), base (580M), large (1.2B), xl (3.7B)

**Vantagem**: Maior capacidade de generalização e melhor performance em línguas de baixo recurso.

### 3.2. Framework Moderno (PyTorch)

**Original**: TensorFlow 2.4
- API mais verbosa
- Debugging mais complexo

**Nossa Implementação**: PyTorch 2.0+
- API mais intuitiva
- Melhor integração com HuggingFace
- Debugging mais fácil
- Comunidade mais ativa

**Vantagem**: Código mais limpo, manutenível e extensível.

### 3.3. Syntax-Aware para Português

**Original**: Tags de dependência alemãs
- Parser spaCy alemão
- Tags específicas do alemão (ep, mo, nk, pd)

**Nossa Implementação**: Tags de dependência portuguesas
- Parser spaCy português (`pt_core_news_lg`)
- 40 tags universais + específicas do português
- Mapeamento: ROOT, nsubj, obj, det, amod, advmod, etc.

**Exemplo**:
```python
Texto: "O menino está jogando bola no parque"
Tokens: ['O', 'menino', 'está', 'jogando', 'bola', 'no', 'parque']
Tags: ['det', 'nsubj', 'aux', 'ROOT', 'obj', 'case', 'obl']
```

### 3.4. Dataset Maior

**Original**: 
- RWTH-PHOENIX-2014-T
- 7,096 amostras de treino
- 519 dev, 642 test
- Domínio: previsão do tempo

**Nossa Implementação**:
- VLibrasBD
- 127,349 pares PT-BR ↔ LIBRAS
- Domínio diverso (governo, geral)
- **18x maior** que o dataset original

**Vantagem**: Mais dados = melhor generalização e robustez.

### 3.5. Transfer Learning Multilíngue

**Original**: Treinamento direto em DGS

**Nossa Implementação**: Estratégia de 3 fases
1. **Fase 1 (Opcional)**: Pré-treinar em ASL + DGS + outras línguas de sinais
2. **Fase 2**: Fine-tune em LIBRAS com syntax-aware embeddings
3. **Fase 3 (Opcional)**: Adaptação de domínio específico

**Vantagem**: Aproveitar conhecimento de outras línguas de sinais.

### 3.6. Back-Translation para Augmentação

**Original**: Sem augmentação de dados

**Nossa Implementação**: Back-translation PT-BR → EN → PT-BR
- Usa modelos MarianMT (Helsinki-NLP)
- Aumenta dataset em 50-100%
- Melhora robustez do modelo

**Exemplo**:
```
Original: "O menino está jogando bola no parque"
Back-translated: "O garoto joga bola no parque"
Gloss (mantida): "MENINO JOGAR BOLA PARQUE"
```

### 3.7. Infraestrutura de Treinamento

**Original**:
- Logging manual
- Checkpointing básico

**Nossa Implementação**:
- **WandB** para tracking de experimentos
- **Mixed Precision (FP16)** para treino mais rápido
- **Gradient Accumulation** para batch sizes maiores
- **Early Stopping** automático
- **Checkpointing** a cada N steps
- **Distributed Training** (suporte multi-GPU)

### 3.8. Testes Unitários

**Original**: Sem testes automatizados

**Nossa Implementação**: Suite completa de testes
- `test_syntax_processor.py`: Testa extração sintática
- `test_tokenizer.py`: Testa tokenização mT5
- `test_metrics.py`: Testa métricas de avaliação

**Comando**: `pytest tests/`

### 3.9. Notebooks de Análise

**Original**: Sem análise exploratória documentada

**Nossa Implementação**:
- `data_exploration.ipynb`: Estatísticas do dataset, distribuições, vocabulário
- `model_analysis.ipynb`: Análise de performance, curvas de aprendizado

### 3.10. Modelo Baseline para Comparação

**Original**: Apenas modelo com sintaxe

**Nossa Implementação**: Dois modelos
1. **Baseline**: mT5 padrão (sem sintaxe)
2. **Syntax-Aware**: mT5 + embeddings sintáticos

**Permite**: Comparação justa do impacto da informação sintática.

---

## 4. COMPARAÇÃO TÉCNICA

| Aspecto | Original (DGS) | Nossa Implementação (LIBRAS) |
|---------|----------------|------------------------------|
| **Framework** | TensorFlow 2.4 | PyTorch 2.0+ |
| **Modelo** | mBART | mT5 (base/large) |
| **Língua Fonte** | Alemão | Português BR |
| **Língua Alvo** | DGS glosses | LIBRAS glosses |
| **Dataset Size** | 7,096 | 127,349 (**18x maior**) |
| **Sintaxe** | spaCy DE | spaCy PT |
| **Transfer Learning** | Não | Sim (multilíngue) |
| **Augmentação** | Não | Sim (back-translation) |
| **Testes** | Não | Sim (pytest) |
| **Notebooks** | Não | Sim (análise completa) |
| **Baseline** | Não | Sim (comparação) |
| **Logging** | Manual | WandB automático |
| **Mixed Precision** | Não | Sim (FP16) |

---

## 5. RESULTADOS ESPERADOS

### Métricas de Avaliação (mesmas do original)
1. **sacreBLEU** ↑ (qualidade de tradução)
2. **TER** ↓ (taxa de erro de edição)
3. **ROUGE-L F1** ↑ (sobreposição de sequências)
4. **METEOR** ↑ (correlação humana)
5. **Generation Time** ↓ (eficiência)

### Hipóteses

**H1**: Modelo syntax-aware supera baseline em BLEU, TER e ROUGE-L
- **Justificativa**: Informação sintática ajuda a modelar transformações gloss

**H2**: Dataset maior (LIBRAS) resulta em melhor generalização
- **Justificativa**: 18x mais dados de treinamento

**H3**: mT5 supera mBART devido a pré-treinamento multilíngue
- **Justificativa**: mT5 tem melhor cobertura linguística

**H4**: Back-translation melhora robustez
- **Justificativa**: Mais variações de entrada

### Resultados do Paper Original (DGS)

| Métrica | Baseline (No-Syntax) | Syntax-Aware | Melhoria |
|---------|----------------------|--------------|----------|
| **sacreBLEU** | ~40.0 | ~45.7 | **+5.7** |
| **TER** | ~0.56 | ~0.41 | **-0.15** |
| **ROUGE-L** | ~0.64 | ~0.65 | **+0.01** |
| **METEOR** | ~0.50 | ~0.47 | **-0.03** |

**Observação**: METEOR foi pior com sintaxe (possível viés da métrica).

### Resultados Esperados (LIBRAS)

Esperamos resultados **iguais ou melhores** devido a:
- Dataset maior
- Modelo mais poderoso (mT5)
- Augmentação de dados
- Transfer learning

---

## 6. ARQUIVOS IMPLEMENTADOS

### Configurações (3)
- `configs/model_config.yaml`: Configuração do modelo
- `configs/training_config.yaml`: Hiperparâmetros
- `configs/data_config.yaml`: Datasets e pré-processamento

### Utilitários (5)
- `utils/syntax_processor.py`: Extração sintática (spaCy PT)
- `utils/tokenizer_utils.py`: Tokenização mT5
- `utils/metrics.py`: Métricas de avaliação
- `utils/data_loader.py`: Carregamento de dados
- `utils/back_translation.py`: Augmentação

### Modelos (1)
- `models/mt5_syntax_aware.py`: MT5SyntaxAware + MT5Baseline

### Scripts (5)
- `scripts/preprocess_data.py`: Pré-processamento
- `scripts/train_mt5_syntax.py`: Treino syntax-aware
- `scripts/train_mt5_baseline.py`: Treino baseline
- `scripts/evaluate.py`: Avaliação
- `scripts/inference.py`: Inferência interativa
- `scripts/demo.py`: Demonstração

### Testes (3)
- `tests/test_syntax_processor.py`
- `tests/test_tokenizer.py`
- `tests/test_metrics.py`

### Notebooks (1)
- `notebooks/data_exploration.ipynb`

**Total**: 18 módulos Python + 3 configs + 1 notebook = **22 arquivos**

---

## 7. COMO EXECUTAR

### Instalação
```bash
cd libras-text2gloss
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
```

### Pré-processamento
```bash
python scripts/preprocess_data.py
```

### Treinamento
```bash
# Baseline
python scripts/train_mt5_baseline.py --debug

# Syntax-Aware
python scripts/train_mt5_syntax.py --debug
```

### Avaliação
```bash
python scripts/evaluate.py \
  --model_syntax models/checkpoints/syntax_aware/best_model \
  --model_baseline models/checkpoints/baseline/best_model
```

### Inferência
```bash
python scripts/inference.py \
  --model models/checkpoints/syntax_aware/best_model \
  --interactive
```

---

## 8. CONCLUSÃO

Este sistema representa uma **evolução significativa** do trabalho original:

✅ **Modelo mais poderoso** (mT5 vs mBART)  
✅ **Dataset 18x maior** (127k vs 7k)  
✅ **Framework moderno** (PyTorch vs TensorFlow)  
✅ **Augmentação de dados** (back-translation)  
✅ **Transfer learning** (multilíngue)  
✅ **Infraestrutura completa** (testes, notebooks, logging)  
✅ **Baseline para comparação**  

O sistema está pronto para:
1. Treinar em dados reais LIBRAS
2. Comparar syntax-aware vs baseline
3. Publicar resultados em paper científico
4. Servir como base para pesquisas futuras em tradução para línguas de sinais
