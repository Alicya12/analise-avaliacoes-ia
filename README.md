# analise-avaliacoes-ia
# 🤖 Projeto: Análise de Sentimento com Machine Learning

Este projeto foi desenvolvido como parte de uma pesquisa acadêmica focada em Processamento de Linguagem Natural (NLP). O objetivo principal é construir e treinar uma Inteligência Artificial capaz de ler avaliações textuais e classificar de forma automatizada o sentimento por trás delas como **positivo** ou **negativo**.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O projeto foi totalmente desenvolvido em **Python 3** utilizando as seguintes ferramentas essenciais:

* **Pandas:** Biblioteca robusta utilizada para a manipulação, limpeza e estruturação de dados em formato de tabelas (DataFrames).
* **Scikit-Learn:** A principal biblioteca de Machine Learning do ecossistema Python, utilizada para:
    * Divisão dos dados em conjuntos de treino e teste (`train_test_split`).
    * Vetorização e processamento estatístico do texto (`TfidfVectorizer`).
    * Treinamento e execução do algoritmo de classificação (`LogisticRegression`).
    * Geração das métricas de desempenho (`accuracy_score`, `classification_report`, `confusion_matrix`).

---

## 📊 Estrutura e Resultados do Modelo

O modelo foi treinado utilizando uma base histórica massiva de avaliações (dividida em 80% para treino do algoritmo e 20% para testes rigorosos de validação).

### 📈 Desempenho Alcançado:
* **Acurácia Geral:** **89,28%** (O modelo classifica corretamente quase 90% das novas frases textuais).
* **F1-Score Equilibrado:** **89%** tanto para críticas positivas quanto negativas, provando que o modelo é estável e não possui viés de classificação.

---

## 🚀 Como Executar o Projeto no GitHub Codespaces

1. Abra o repositório no ambiente virtual do Codespaces.
2. Instale as dependências listadas no projeto utilizando o terminal:
   ```bash
   pip install -r requirements.txt