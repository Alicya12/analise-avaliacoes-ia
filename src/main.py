import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("🤖 [1/4] Carregando o conjunto de dados...")
df = pd.read_csv("dados/dataset.csv")

print(df.head())

X = df['review']       
y = df['sentiment']    

print("✂️ [2/4] Dividindo os dados em Treino (80%) e Teste (20%)...")
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.20, random_state=42)

print("🔤 [3/4] Convertendo textos em números (Vetorização TF-IDF)...")
vetorizador = TfidfVectorizer(max_features=5000, stop_words='english')
X_treino_vetorizado = vetorizador.fit_transform(X_treino)
X_teste_vetorizado = vetorizador.transform(X_teste)

print("🏋️‍♂️ [4/4] Treinando o modelo de Machine Learning (Regressão Logística)...")
modelo = LogisticRegression()
modelo.fit(X_treino_vetorizado, y_treino)

print("\n📊 --- RESULTADOS E MÉTRICAS DE DESEMPENHO ---")
predicoes = modelo.predict(X_teste_vetorizado)

acuracia = accuracy_score(y_teste, predicoes)
print(f"Acurácia Geral do Modelo: {acuracia:.4f} ({acuracia * 100:.2f}%)")

print("\n📋 Relatório de Classificação Completo (Precisão, Recall, F1-Score):")
print(classification_report(y_teste, predicoes))

print("\n🧩 Matriz de Confusão:")
print(confusion_matrix(y_teste, predicoes))