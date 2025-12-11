# Guia de Compilação LaTeX - Artigo Científico

## 📄 Arquivos Criados

1. **artigo.tex** - Documento LaTeX principal
2. **referencias.bib** - Arquivo de bibliografia BibTeX

## 🔧 Pré-requisitos

Para compilar o documento LaTeX, você precisa:

1. **LaTeX Distribution**:
   - TeX Live (recomendado)
   - MiKTeX (Windows)
   - MacTeX (macOS)

2. **Template SBC**:
   - Baixar `sbc-template.cls` e `sbc-template.bst` do site da SBC
   - Colocar na mesma pasta do `artigo.tex`

3. **Pacotes Necessários**:
   - `graphicx` (para figuras)
   - `url` (para URLs)
   - `amsmath` (para fórmulas matemáticas)
   - `booktabs` (para tabelas profissionais)
   - `multirow` (para células múltiplas em tabelas)

## 📝 Como Compilar

### Método 1: Compilação Manual

```bash
# 1. Compilar LaTeX
pdflatex artigo.tex

# 2. Compilar bibliografia (se usar .bib)
bibtex artigo

# 3. Compilar novamente (2x para referências)
pdflatex artigo.tex
pdflatex artigo.tex
```

### Método 2: Compilação Automática

```bash
# Compilar tudo de uma vez
latexmk -pdf artigo.tex
```

### Método 3: Overleaf (Online)

1. Acesse [overleaf.com](https://www.overleaf.com)
2. Crie novo projeto
3. Faça upload dos arquivos:
   - `artigo.tex`
   - `referencias.bib`
   - `sbc-template.cls` (template SBC)
   - `sbc-template.bst` (bibliografia SBC)
4. Compile automaticamente

## 📋 Estrutura do Documento

O artigo LaTeX contém:

- **Abstract** (inglês)
- **Resumo** (português)
- **1. Introdução** (Motivação, Desafio, Contribuições)
- **2. Trabalhos Relacionados**
- **3. Metodologia** (Dataset, Arquitetura, Hiperparâmetros)
- **4. Experimentos e Resultados**
- **5. Discussão**
- **6. Conclusões e Trabalhos Futuros**
- **Referências Bibliográficas**

## 🔍 Personalização Necessária

Antes de compilar, você precisa preencher:

1. **Autores** (linha ~20):
   ```latex
   \author{[Seus Nomes]\inst{1}, [Nome 2]\inst{2}}
   ```

2. **Afiliações** (linha ~23):
   ```latex
   \address{[Sua Instituição]\\
     [Endereço Completo]\\
     [Cidade] -- [Estado] -- [País]
   ```

3. **Emails** (linha ~29):
   ```latex
   \email{[email1@instituicao.br], [email2@instituicao.br]}
   ```

4. **Agradecimentos** (seção opcional):
   ```latex
   \section*{Agradecimentos}
   Os autores agradecem [agradecimentos específicos].
   ```

## 📊 Tabelas e Figuras

O documento inclui 4 tabelas:
- Tabela 1: Particionamento do Dataset
- Tabela 2: Comparação de Datasets
- Tabela 3: Hiperparâmetros
- Tabela 4: Resultados Esperados

**Nota**: Se você tiver figuras, adicione-as na pasta e referencie:
```latex
\includegraphics[width=.5\textwidth]{figura.png}
```

## ⚠️ Problemas Comuns

### Erro: "sbc-template.cls not found"
**Solução**: Baixe o template SBC e coloque na mesma pasta

### Erro: "Package babel Error"
**Solução**: Instale o pacote babel ou remova `\usepackage[brazil]{babel}`

### Erro: "Undefined citation"
**Solução**: Compile com `bibtex` e depois `pdflatex` novamente (2x)

### Erro: "Overfull hbox"
**Solução**: Ajuste texto ou use `\sloppy` (já incluído)

## 📝 Notas Importantes

1. **Template SBC**: O template oficial da SBC deve ser baixado do site da conferência
2. **Referências**: As referências estão no formato SBC (arquivo .bib)
3. **Formatação**: O documento segue o padrão SBC (A4, margens específicas)
4. **Idioma**: Abstract em inglês, Resumo em português (conforme padrão SBC)

## 🎯 Próximos Passos

1. Preencher informações dos autores
2. Adicionar figuras (se houver)
3. Revisar conteúdo
4. Compilar e verificar PDF
5. Submeter para conferência

---

**Última atualização**: Após criação do documento LaTeX


