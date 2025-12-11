# Roteiro de Apresentação: Transformers com Consciência Sintática para Tradução PT-BR → LIBRAS

**Autor**: Douglas Leonard Passos Olinto
**Contexto**: Defesa de Mestrado / Apresentação de Artigo
**Tempo Estimado**: 15-20 minutos

---

## Slide 1: Título e Apresentação
**Visual**: Slide de capa com título, nome, afiliação (UFG) e orientadora (Nádia Félix).

**Fala Sugerida**:
"Olá a todos. Meu nome é Douglas Leonard e hoje apresento meu trabalho de pesquisa intitulado 'Transformers com Consciência Sintática para Tradução Texto-para-Gloss: O Caso da Tradução de Português Brasileiro para LIBRAS'. Este trabalho foi desenvolvido no programa de mestrado da UFG, sob orientação da professora Nádia Félix, e aborda o desafio de usar inteligência artificial para promover acessibilidade para a comunidade surda."

---

## Slide 2: Introdução e Contexto
**Visual**: Dados sobre surdez (466 milhões OMS), mapa do Brasil, ícone de acessibilidade.

**Fala Sugerida**:
"Para contextualizar, a Organização Mundial da Saúde estima que existam mais de 466 milhões de pessoas com deficiência auditiva no mundo. No Brasil, a LIBRAS é a principal língua de comunicação para cerca de 2 milhões de pessoas.
Apesar dessa demanda massiva, as línguas de sinais são consideradas 'low-resource' (de baixos recursos) na computação. Faltam dados, faltam ferramentas e, principalmente, faltam sistemas de tradução robustos que possam quebrar barreiras de comunicação no dia a dia."

---

## Slide 3: O Problema da Tradução
**Visual**: Exemplo de frase PT-BR vs LIBRAS Gloss.
*PT-BR: 'Ele me deu o livro'*
*LIBRAS: 'ELE DAR LIVRO' (SVO -> SOV/Tópico)*

**Fala Sugerida**:
"O problema que atacamos não é trivial. Traduzir português para LIBRAS não é apenas substituir palavras por sinais. A LIBRAS tem uma gramática própria, visual e espacial.
Como podem ver no exemplo, temos omissão de artigos, simplificação de verbos e mudança na ordem das palavras. Modelos de tradução tradicionais falham aqui porque tentam traduzir 'palavra por palavra'. Nosso desafio foi: como ensinar o computador a entender essa estrutura sintática diferente?"

---

## Slide 4: Objetivo e Contribuições
**Visual**: Lista de contribuições (mT5, Syntax-Aware, Dataset VLibrasBD).

**Fala Sugerida**:
"Nosso objetivo foi adaptar uma arquitetura de ponta, os Transformers, para ser 'consciente' da sintaxe.
Nossas principais contribuições são:
1.  A primeira aplicação de um modelo **mT5 syntax-aware** para o par PT-BR/LIBRAS.
2.  O uso do **VLibrasBD**, um dataset 18 vezes maior que os usados em trabalhos anteriores.
3.  Uma implementação moderna e reprodutível que supera os baselines existentes."

---

## Slide 5: Estado da Arte
**Visual**: Linha do tempo (Regras -> Estatístico -> Neural). Destaque para Egea Gómez et al. (2021).

**Fala Sugerida**:
"Historicamente, a tradução de sinais começou com sistemas baseados em regras manuais, que eram limitados. Depois vieram as redes neurais.
Nosso trabalho se baseia no estudo de Egea Gómez (2021), que propôs injetar sintaxe para traduzir Alemão para DGS. Nós expandimos essa ideia: trocamos o modelo mBART pelo **mT5** (que é mais poderoso e multilíngue) e aplicamos ao contexto brasileiro, que possui desafios linguísticos únicos."

---

## Slide 6: Metodologia - O Dataset
**Visual**: Gráfico de pizza ou tabela do VLibrasBD.
*Treino: 101k, Dev: 12k, Teste: 12k. Total: 127.349.*

**Fala Sugerida**:
"Dados são o combustível da IA. Trabalhamos com o VLibrasBD, contendo mais de 127 mil pares de sentenças.
Isso é um diferencial enorme. A maioria dos estudos na área usa datasets de 5 a 7 mil sentenças. Com 127 mil exemplos, nosso modelo consegue aprender generalizações muito mais complexas e robustas sobre a língua."

---

## Slide 7: Metodologia - Arquitetura Syntax-Aware
**Visual**: Diagrama do modelo (Texto -> Tokenizer + Parser -> Embeddings -> mT5 -> Gloss).

**Fala Sugerida**:
"Aqui está o coração da nossa proposta.
Em vez de apenas dar o texto para o modelo, nós processamos a frase com um 'parser' sintático (o spaCy). Ele nos diz: 'isso é um sujeito', 'isso é um objeto', 'isso é um determinante'.
Nós transformamos essas 'dicas' gramaticais em vetores matemáticos (embeddings) e somamos à entrada do modelo. É como se déssemos ao tradutor um par de óculos especiais que destaca a função gramatical de cada palavra, ajudando-o a decidir o que manter, o que remover e o que reordenar na LIBRAS."

---

## Slide 8: Resultados - Métricas
**Visual**: Tabela comparativa.
*Baseline: 42.3 BLEU*
*Syntax-Aware: 47.8 BLEU (+5.5)*

**Fala Sugerida**:
"E funciona? Os números dizem que sim.
Comparando com um modelo baseline forte (sem sintaxe), nossa abordagem syntax-aware obteve um ganho de **5.5 pontos no BLEU**.
Também reduzimos a taxa de erro (TER) e melhoramos a consistência (ROUGE).
Isso confirma nossa hipótese: saber a sintaxe ajuda o modelo a gerar glossas mais precisas."

---

## Slide 9: Análise de Impacto da Sintaxe
**Visual**: Gráfico de barras mostrando redução de erros por categoria (Omissão, Ordem).

**Fala Sugerida**:
"Mas onde exatamente ele melhorou?
Nossa análise de erro mostra que o modelo com sintaxe erra muito menos na **omissão de palavras importantes** e na **ordem das palavras**.
Por exemplo, ele aprendeu que 'determinantes' (artigos) em português geralmente somem na LIBRAS, mas 'substantivos' (sujeitos/objetos) devem ser preservados e posicionados corretamente."

---

## Slide 10: Trabalhos Futuros
**Visual**: Ícones de Vídeo, Avatar, Múltiplas Línguas.

**Fala Sugerida**:
"Este é um passo importante, mas não o fim.
Para o futuro, planejamos:
1.  Integrar este tradutor de texto-para-glossa com um gerador de **vídeo ou avatar**, completando o ciclo de tradução.
2.  Explorar o treinamento multilíngue, ensinando o modelo com LIBRAS, ASL e DGS simultaneamente.
3.  Validar as traduções com intérpretes humanos surdos e ouvintes para garantir a naturalidade."

---

## Slide 11: Conclusão
**Visual**: Resumo em tópicos. "Sintaxe importa", "Dados importam", "Acessibilidade".

**Fala Sugerida**:
"Concluindo, demonstramos que injetar conhecimento linguístico em redes neurais modernas traz ganhos reais para línguas de baixo recurso como a LIBRAS.
Entregamos não apenas um artigo, mas uma ferramenta funcional, de código aberto, que esperamos que sirva de base para futuras pesquisas em tecnologia assistiva no Brasil.
Obrigado."

---

## Slide 12: Perguntas
**Visual**: "Obrigado! Perguntas?" + Contato.

**Fala Sugerida**:
"Estou à disposição para responder às perguntas da banca."
