# Plano de Ação - Melhorias para Atender Critérios de Avaliação

**Baseado em**: Critérios de Avaliação do Projeto Final - PLN 2025/2  
**Status Atual**: 85-95% (51-57/60 pontos)  
**Meta**: 95-100% (57-60/60 pontos)

---

## 🎯 RESUMO EXECUTIVO

### Pontuação Atual por Critério

| Critério | Peso | Atual | Meta | Gap |
|----------|------|-------|------|-----|
| 1. Clareza no problema | 10% | 9-10/10 ✅ | 10/10 | 0-1 |
| 2. Revisão de literatura | 10% | 9-10/10 ✅ | 10/10 | 0-1 |
| 3. Metodologia | 15% | 14-15/15 ✅ | 15/15 | 0-1 |
| 4. Experimentação | 15% | 10-12/15 ⚠️ | 14-15/15 | 2-5 |
| 5. Redação e estrutura | 10% | 9-10/10 ✅ | 10/10 | 0-1 |
| **TOTAL** | **60%** | **51-57/60** | **57-60/60** | **2-9** |

---

## 🔴 AÇÕES CRÍTICAS (Antes da Entrega)

### 1. Treinar Modelos e Obter Resultados Reais

**Problema**: Resultados são esperados (literatura), não empíricos  
**Impacto**: -3 a -5 pontos em Experimentação (15%)  
**Prioridade**: 🔴 CRÍTICA

**Ações**:
```bash
# 1. Obter dataset completo (ou subset maior)
# 2. Treinar modelo baseline
python scripts/train_mt5_baseline.py \
  --config configs/training_config.yaml \
  --output models/checkpoints/baseline \
  --debug  # Para teste rápido

# 3. Treinar modelo syntax-aware
python scripts/train_mt5_syntax.py \
  --config configs/training_config.yaml \
  --output models/checkpoints/syntax_aware \
  --debug  # Para teste rápido

# 4. Avaliar ambos os modelos
python scripts/evaluate.py \
  --model_baseline models/checkpoints/baseline/best_model \
  --model_syntax models/checkpoints/syntax_aware/best_model \
  --test_data data/processed/test.json
```

**Resultado Esperado**:
- Tabela com resultados reais (não esperados)
- Comparação empírica Baseline vs Syntax-aware
- Análise estatística (se possível)

**Tempo Estimado**: 2-4 horas (com GPU) ou 8-12 horas (CPU)

---

### 2. Atualizar Seção de Resultados no LaTeX

**Ações**:
1. Substituir "Resultados Esperados" por "Resultados Obtidos"
2. Adicionar tabela com resultados reais
3. Incluir análise estatística
4. Adicionar exemplos de tradução

**Código LaTeX a Adicionar**:
```latex
\subsection{Resultados Obtidos}

Após treinamento completo dos modelos, obtivemos os seguintes resultados:

\begin{table}[ht]
\centering
\caption{Resultados Obtidos: Baseline vs Syntax-Aware}
\label{tab:resultados_obtidos}
\begin{tabular}{lccc}
\toprule
Métrica & Baseline & Syntax-Aware & Melhoria \\
\midrule
sacreBLEU ↑ & [VALOR REAL] & [VALOR REAL] & [MELHORIA] \\
TER ↓ & [VALOR REAL] & [VALOR REAL] & [MELHORIA] \\
ROUGE-L F1 ↑ & [VALOR REAL] & [VALOR REAL] & [MELHORIA] \\
METEOR ↑ & [VALOR REAL] & [VALOR REAL] & [MELHORIA] \\
\bottomrule
\end{tabular}
\end{table}
```

**Tempo Estimado**: 30 minutos

---

## 🟡 AÇÕES IMPORTANTES (Melhora Significativa)

### 3. Adicionar Figuras e Gráficos

**Problema**: Falta de visualizações  
**Impacto**: -1 a -2 pontos em Redação (10%)  
**Prioridade**: 🟡 IMPORTANTE

**Figuras a Criar**:

1. **Diagrama da Arquitetura**:
   - Fluxo: Input → Parser → Tokenizer → Modelo → Output
   - Mostrar injeção de syntax embeddings
   - Formato: PNG/SVG, preto e branco

2. **Pipeline de Processamento**:
   - Etapas de pré-processamento
   - Fluxo de dados
   - Formato: Diagrama de blocos

3. **Gráfico de Curvas de Aprendizado** (se tiver treinamento):
   - Loss ao longo das épocas
   - BLEU no dev set
   - Comparação Baseline vs Syntax-aware

4. **Análise de Erros**:
   - Tipos de erro mais comuns
   - Distribuição de erros por categoria
   - Gráfico de barras ou pizza

**Tempo Estimado**: 2-3 horas

---

### 4. Expandir Análise de Resultados

**Ações**:
1. Adicionar exemplos de tradução (sucesso e erro)
2. Análise qualitativa detalhada
3. Discussão de casos específicos
4. Comparação com trabalho original mais detalhada

**Código LaTeX a Adicionar**:
```latex
\subsection{Análise Qualitativa}

Apresentamos exemplos de tradução para análise qualitativa:

\begin{table}[ht]
\centering
\caption{Exemplos de Tradução}
\label{tab:exemplos}
\begin{tabular}{p{4cm}p{4cm}p{4cm}}
\toprule
PT-BR & LIBRAS (Referência) & Syntax-Aware (Predito) \\
\midrule
O menino está jogando bola no parque. & MENINO JOGAR BOLA PARQUE & MENINO JOGAR BOLA PARQUE ✓ \\
A professora ensina matemática para os alunos. & PROFESSORA ENSINAR MATEMATICA ALUNO & PROFESSORA ENSINAR MATEMATICA ALUNO ✓ \\
\bottomrule
\end{tabular}
\end{table}
```

**Tempo Estimado**: 1-2 horas

---

## 🟢 AÇÕES OPCIONAIS (Melhora Incremental)

### 5. Detalhar Pré-processamento

**Ações**:
- Adicionar seção específica sobre pré-processamento
- Estatísticas do dataset
- Normalização aplicada
- Limpeza de dados

**Tempo Estimado**: 1 hora

### 6. Adicionar Ablation Studies

**Ações**:
- Impacto de cada componente
- Análise de diferentes tags sintáticas
- Tamanho do modelo vs performance

**Tempo Estimado**: 2-3 horas (requer treinamentos adicionais)

---

## 📋 CHECKLIST DE MELHORIAS

### 🔴 Crítico (Fazer Antes da Entrega)

- [ ] **Treinar modelo baseline** em dataset completo (ou subset maior)
- [ ] **Treinar modelo syntax-aware** em dataset completo
- [ ] **Avaliar ambos os modelos** e obter resultados reais
- [ ] **Atualizar seção de resultados** no LaTeX com dados empíricos
- [ ] **Substituir "esperados" por "obtidos"** em todas as tabelas

### 🟡 Importante (Melhora Significativa)

- [ ] **Criar diagrama da arquitetura** (figura)
- [ ] **Criar pipeline de processamento** (figura)
- [ ] **Adicionar gráficos de curvas de aprendizado** (se tiver treinamento)
- [ ] **Adicionar análise de erros** (gráfico)
- [ ] **Expandir exemplos de tradução** (tabela)

### 🟢 Opcional (Melhora Incremental)

- [ ] Detalhar pré-processamento
- [ ] Adicionar ablation studies
- [ ] Incluir mais trabalhos recentes na revisão
- [ ] Adicionar pseudocódigo da arquitetura

---

## ⏱️ CRONOGRAMA SUGERIDO

### Semana 1 (Crítico)

**Dia 1-2**: Treinamento de Modelos
- Obter/preparar dataset
- Treinar baseline
- Treinar syntax-aware

**Dia 3**: Avaliação e Análise
- Avaliar modelos
- Calcular métricas
- Análise de resultados

**Dia 4**: Atualização do Artigo
- Atualizar seção de resultados
- Substituir dados esperados por obtidos
- Revisar texto

### Semana 2 (Importante)

**Dia 1-2**: Criação de Figuras
- Diagrama arquitetura
- Pipeline processamento
- Gráficos (se tiver dados)

**Dia 3**: Expansão de Análise
- Exemplos de tradução
- Análise qualitativa
- Discussão detalhada

**Dia 4**: Revisão Final
- Revisar todo o artigo
- Verificar formatação
- Preparar apresentação

---

## 📊 PROJEÇÃO DE PONTUAÇÃO

### Cenário Atual (Sem Melhorias)
- **Artigo**: 51-57/60 (85-95%)
- **Apresentação**: 0/40 (não criada)
- **TOTAL**: 51-57/100 (51-57%)

### Cenário com Melhorias Críticas
- **Artigo**: 54-58/60 (90-97%)
- **Apresentação**: 30-35/40 (estimado)
- **TOTAL**: 84-93/100 (84-93%)

### Cenário com Todas as Melhorias
- **Artigo**: 57-60/60 (95-100%)
- **Apresentação**: 35-40/40 (estimado)
- **TOTAL**: 92-100/100 (92-100%)

---

## 🎯 PRIORIZAÇÃO

### Fazer AGORA (Crítico)
1. ✅ Treinar modelos e obter resultados reais
2. ✅ Atualizar seção de resultados no LaTeX

### Fazer DEPOIS (Importante)
3. ✅ Criar figuras e gráficos
4. ✅ Expandir análise de resultados

### Fazer SE SOBRAR TEMPO (Opcional)
5. ⏳ Detalhar pré-processamento
6. ⏳ Adicionar ablation studies

---

## ✅ CONCLUSÃO

O artigo está **bem estruturado e tecnicamente sólido**, mas precisa de **resultados empíricos reais** para atingir pontuação máxima.

**Ação Principal**: Treinar modelos e obter resultados reais (gap de 2-5 pontos)

**Tempo Total Estimado**: 6-10 horas para melhorias críticas e importantes

---

**Última atualização**: Após análise dos critérios de avaliação


