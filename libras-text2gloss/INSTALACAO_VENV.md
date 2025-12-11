# Guia de Instalação do Ambiente Virtual

## ⚠️ Problema Encontrado

Durante a instalação, foi detectado **falta de espaço em disco** ao tentar instalar PyTorch com suporte CUDA (requer ~2GB).

## ✅ Soluções

### Opção 1: Instalar PyTorch CPU (Recomendado se não tiver GPU)

```bash
cd libras-text2gloss
source venv/bin/activate

# Instalar PyTorch CPU (muito menor, ~200MB)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar resto das dependências
pip install -r requirements.txt
```

### Opção 2: Limpar Espaço e Instalar Completo

1. **Verificar espaço disponível**:
```bash
df -h
```

2. **Limpar cache do pip**:
```bash
pip cache purge
```

3. **Limpar pacotes não utilizados**:
```bash
pip autoremove
```

4. **Instalar novamente**:
```bash
pip install -r requirements.txt
```

### Opção 3: Instalar Dependências Essenciais Primeiro

```bash
cd libras-text2gloss
source venv/bin/activate

# Instalar apenas o essencial
pip install torch transformers spacy nltk pandas numpy pyyaml tqdm

# Depois instalar o resto conforme necessário
pip install sacrebleu rouge-score datasets accelerate sentencepiece
pip install wandb matplotlib seaborn pytest black flake8 jupyter
```

## 📋 Status da Instalação

### ✅ Já Instalado (antes do erro)
- pip, setuptools, wheel atualizados
- Dependências básicas baixadas

### ❌ Falhou
- PyTorch com CUDA (falta de espaço)
- Dependências finais

## 🔧 Comandos Úteis

### Ativar ambiente virtual
```bash
cd libras-text2gloss
source venv/bin/activate
```

### Verificar instalação
```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import spacy; print(spacy.__version__)"
```

### Desativar ambiente virtual
```bash
deactivate
```

### Remover ambiente virtual
```bash
rm -rf venv
```

## 📝 Notas

- **PyTer**: Foi comentado no requirements.txt pois tem dependências problemáticas. O sacrebleu já inclui TER (Translation Edit Rate).
- **Espaço necessário**: ~2-3GB para instalação completa com CUDA, ~500MB para CPU-only.
- **Tempo de instalação**: 5-15 minutos dependendo da conexão.

## 🚀 Próximos Passos Após Instalação

1. **Baixar modelo spaCy português**:
```bash
python -m spacy download pt_core_news_lg
```

2. **Baixar dados NLTK**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

3. **Testar instalação**:
```bash
python -c "from models.mt5_syntax_aware import MT5SyntaxAware; print('OK')"
```

## 💡 Recomendações

- Se você **tem GPU NVIDIA**: Use Opção 2 (limpar espaço e instalar completo)
- Se você **não tem GPU** ou **tem pouco espaço**: Use Opção 1 (PyTorch CPU)
- Para **desenvolvimento/testes**: Use Opção 3 (instalação mínima)

---

**Última atualização**: Após tentativa de instalação completa

