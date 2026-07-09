# Dio_Deteccao_de_Anomalias_em_Transacoes_em_Python-
Detecção de Anomalias em Transações em Python
# Detecção de Anomalias em Transações Financeiras

Este projeto demonstra a aplicação de técnicas de Machine Learning para a detecção de fraudes em transações financeiras, utilizando um conjunto de dados de transações com features anonimizadas.

## Propósito

O objetivo principal é construir e avaliar modelos capazes de identificar transações fraudulentas, um problema desafiador devido à natureza desbalanceada dos dados (poucas fraudes em comparação com transações normais). Exploramos diferentes abordagens para lidar com o desbalanceamento e otimizar a performance dos modelos na detecção de fraudes.

## Estrutura do Código

O notebook está organizado nas seguintes etapas:

1.  **Carregamento e Exploração Inicial dos Dados:**
    *   Carrega o dataset `creditcard.csv` usando `pandas`.
    *   Realiza uma análise inicial da distribuição da variável `Class` (0 para transações normais, 1 para fraudes), evidenciando o grande desbalanceamento.

2.  **Pré-processamento e Engenharia de Features:**
    *   A coluna `Amount` é transformada usando logaritmo (`np.log1p`) para reduzir a assimetria.
    *   A coluna `Amount` também é padronizada (`StandardScaler`) para garantir que o modelo não seja indevidamente influenciado pela magnitude dos valores.

3.  **Divisão dos Dados:**
    *   O dataset é dividido em conjuntos de treino e teste (`x_train`, `x_test`, `y_train`, `y_test`) usando `train_test_split` com `stratify=y` para manter a proporção das classes original em ambos os conjuntos.

4.  **Modelagem e Avaliação (Regressão Logística):**
    *   Um modelo de Regressão Logística é treinado nos dados.
    *   A performance é avaliada usando um `classification_report`, que inclui precisão, recall, f1-score e suporte.
    *   Curvas ROC (Receiver Operating Characteristic) e Precision-Recall são geradas, juntamente com o cálculo da AUC (Area Under the Curve) para uma avaliação mais robusta em dados desbalanceados.
    *   Uma análise com um limiar de classificação customizado (0.3) é realizada para observar o impacto nas métricas.

5.  **Técnicas de Balanceamento (Demonstrações):**
    *   **Undersampling:** Um conjunto de dados balanceado é criado por amostragem aleatória de transações normais para igualar o número de fraudes.
    *   **Oversampling (SMOTE):** A técnica SMOTE (`imblearn.over_sampling.SMOTE`) é utilizada para gerar amostras sintéticas da classe minoritária, buscando balancear o dataset. (Nota: Embora demonstrado, os modelos subsequentes foram treinados nos dados `x_train`, `y_train` originais sem aplicar SMOTE diretamente).

6.  **Outros Modelos de Classificação:**
    *   **Random Forest Classifier:** Um modelo Random Forest é treinado com `class_weight='balanced'` para lidar com o desbalanceamento, e sua performance é avaliada.
    *   **XGBoost Classifier:** Um modelo XGBoost é treinado, utilizando `scale_pos_weight` para dar mais peso à classe minoritária. Sua performance é avaliada, e a importância das features é visualizada.

7.  **Otimização de Hiperparâmetros:**
    *   `GridSearchCV` é usado para encontrar os melhores hiperparâmetros (`max_depth`, `n_estimators`) para o modelo XGBoost, otimizando o `recall`.

8.  **Interpretabilidade do Modelo (SHAP):**
    *   A biblioteca `shap` é utilizada para explicar as previsões do modelo XGBoost, mostrando a contribuição de cada feature para as decisões do modelo.

## Bibliotecas Utilizadas

*   `pandas`: Manipulação e análise de dados.
*   `numpy`: Operações numéricas.
*   `sklearn`: Ferramentas de Machine Learning (pré-processamento, modelos, métricas).
*   `matplotlib.pyplot`: Geração de gráficos e visualizações.
*   `imblearn.over_sampling.SMOTE`: Técnica para balanceamento de classes.
*   `xgboost`: Implementação do algoritmo Gradient Boosting.
*   `shap`: Para interpretabilidade de modelos de Machine Learning.

## Resultados Chave

Os resultados demonstram que modelos como Random Forest e XGBoost, especialmente com técnicas para lidar com o desbalanceamento de classes (como `class_weight` ou `scale_pos_weight`), conseguem um `recall` significativamente maior para a classe minoritária (fraudes) em comparação com a Regressão Logística sem ajustes específicos, mantendo uma alta precisão para a classe majoritária (transações normais). A otimização de hiperparâmetros e a análise SHAP contribuem para a melhoria do modelo e para o entendimento de seu comportamento.

---
