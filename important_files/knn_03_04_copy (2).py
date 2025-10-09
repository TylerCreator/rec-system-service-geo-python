import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, ndcg_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

import json
import time

# Загрузка данных
df_calls = pd.read_csv('calls.csv', sep=';')
mid_unique = df_calls['mid'].unique() #  уникальные сервисы, всего 210 сервисов
owner_unique = df_calls['owner'].unique() # уникальные пользователи, всего 152 пользователя
# print("unique", mid_unique)
# print("owner_unique", owner_unique)

# Разбиение на тренировочную (70%) и тестовую (30%) выборки по времени
df_calls = df_calls.sort_values('start_time')
train_size = int(df_calls.shape[0] * 0.7)
df_train = df_calls[:train_size]
df_test = df_calls[train_size:]


# Функция подготовки матрицы
def prepare_df(df, mid_unique, owner_unique):
    pivot = df.pivot_table(values='id', index='owner', columns='mid', aggfunc='count').fillna(0)
    res = np.zeros((owner_unique.shape[0], mid_unique.shape[0]))
    for i in range(len(owner_unique)):
        for j in range(len(mid_unique)):
            res[i][j] = pivot.loc[owner_unique[i], mid_unique[j]] if owner_unique[i] in pivot.index and mid_unique[
                j] in pivot.columns else 0
        s = np.sum(res[i])
        if s > 0:
            res[i] /= s
    return res


# Подготовка матриц
X_train = prepare_df(df_train, mid_unique, owner_unique)
X_test = prepare_df(df_test, mid_unique, owner_unique) # какой пользователь использовал какой сервис

# for als
confidence = 20.0
X_train_sparse = csr_matrix(X_train) * confidence

# Вычисление популярности сервисов (на основе тренировочной выборки)
def get_popular_services(df, mid_unique):
    service_counts = df['mid'].value_counts().reindex(mid_unique, fill_value=0)
    return np.argsort(service_counts)[::-1]  # Индексы сервисов в порядке убывания популярности


popular_services = get_popular_services(df_train, mid_unique)
# print("popular_services", popular_services)

# Класс KNN
class MyNN:
    def __init__(self, n_neighbors=4, metric='cosine'):
        self.n_neighbors = n_neighbors + 1
        self.metric = metric

    def fit(self, X):
        self.X = X
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(X)
        self.distances, self.indices = self.nbrs.kneighbors(X)
        return self

    def predict(self, X):
        preds = np.zeros(X.shape)
        for i in range(self.indices.shape[0]):
            temp = np.zeros(X[0].shape)
            for neighbor in self.indices[i][1:]:
                temp += self.X[neighbor]
            temp /= (self.n_neighbors - 1)
            preds[i] = temp
        return preds

# TSVD модель
class MyTSVD:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components)

    def fit(self, X):
        self.X = X
        self.svd.fit(X)
        self.svd_matrix = self.svd.transform(X)
        return self

    def predict(self, X):
        return self.svd.inverse_transform(self.svd_matrix)

# SVD модель
class MySVD:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, svd_solver="arpack")

    def fit(self, X):
        self.X = X
        self.pca_matrix = self.pca.fit_transform(X)
        return self

    def predict(self, X):
        return self.pca.inverse_transform(self.pca_matrix)

class MyMF:
    def __init__(self, mf):
        self.mf = mf

    def fit(self, X):
        self.X = X
        self.matrix = self.mf.fit_transform(X)
        return self

    def predict(self, X):
        return self.mf.inverse_transform(self.matrix)

# Обучение и предсказание
knn = MyNN(n_neighbors=4).fit(X_train)
preds = knn.predict(X_train)

# for svd
# svd = MyTSVD(n_components=20).fit(X_train)
# preds = svd.predict(X_train)

#NFM
#NMF(n_components=n_components, init='random', random_state=0)

# for als
model = AlternatingLeastSquares(factors=20, iterations=100, regularization=0.1)
model.fit(X_train_sparse)
preds = np.dot(model.user_factors, model.item_factors.T)


# Получение использованных сервисов
def get_used_services(owner_idx, X):
    return np.where(X[owner_idx] > 1e-10)[0]


# Генерация рекомендаций с учетом популярности и исключения использованных сервисов
def get_recommendations(owner_idx, preds, used_services, popular_services, k=10):
    user_preds = preds[owner_idx]
    sorted_services = np.argsort(user_preds)[::-1]  # Сортировка по предсказаниям
    unused_services = [s for s in sorted_services if s not in used_services]  # Исключаем использованные

    # Если предсказанных сервисов недостаточно, дополняем популярными
    if len(unused_services) < k:
        popular_unused = [s for s in popular_services if s not in used_services and s not in unused_services]
        unused_services.extend(popular_unused)

    return unused_services[:k]


# funciton for evaluating knn, als, svd
def calculate_metrics(X_test, preds, popular_services, k=5):
    precisions = []
    recalls = []
    ndcgs = []
    accuracies = []

    for i in range(len(owner_unique)):
        actual = set(np.where(X_test[i] > 1e-10)[0])  # Фактические сервисы в тесте
        # print(i, "actual", actual)
        if not actual:
            continue

        used = set(get_used_services(i, X_train))  # Использованные в тренировке, не учитываем их Если X_train[i] = [0.2, 0, 0.8, 0, 0], то used = {0, 2}
        predicted = get_recommendations(i, preds, used, popular_services, k)  # Рекомендации
        # print(i,"predicted", predicted)

        # Бинарные метки для precision и recall
        y_true = [1 if j in actual else 0 for j in range(len(mid_unique))] # список длиной len(mid_unique) (количество всех сервисов), где 1 — если сервис был в actual (использован в тесте), 0 — если нет
        y_pred = [1 if j in predicted else 0 for j in range(len(mid_unique))] # список той же длины, где 1 — если сервис попал в predicted (рекомендован), 0 — если нет.

        # Relevance scores для NDCG (используем популярность как релевантность)
        y_true_rel = [1 if j in actual else 0 for j in predicted] #список длиной k (длина predicted), где 1 — если рекомендованный сервис был в actual, 0 — если нет. Это "фактическая релевантность".
        y_pred_rel = [preds[i][j] if j in predicted and preds[i][j] > 0 else 1 / len(mid_unique) for j in predicted]

        precisions.append(precision_score(y_true, y_pred)) # precision_score(y_true, y_pred) — доля рекомендованных сервисов, которые были в actual (TP / (TP + FP)).
        recalls.append(recall_score(y_true, y_pred)) # recall_score(y_true, y_pred) — доля сервисов из actual, которые попали в рекомендации (TP / (TP + FN)).
        accuracies.append(accuracy_score(y_true, y_pred))
        if y_true_rel and y_pred_rel:
            ndcgs.append(ndcg_score([y_true_rel], [y_pred_rel])) # ndcg_score([y_true_rel], [y_pred_rel]) — нормализованная дисконтированная кумулятивная выгода, учитывающая порядок рекомендаций. Вычисляется только если списки не пустые.




    return {
        'precision@k': np.mean(precisions),
        'recall@k': np.mean(recalls),
        'ndcg@k': np.mean(ndcgs) if ndcgs else 0,
        'accuracy@k': np.mean(accuracies)  # вернём accuracy
    }


def get_most_popular_service_test(owner_idx, X_test):
    user_test_services = np.where(X_test[owner_idx] > 1e-10)[0]

    if len(user_test_services) == 0:
        return None, []

    service_scores = X_test[owner_idx][user_test_services]
    most_popular_idx = np.argmax(service_scores)
    most_popular_service = user_test_services[most_popular_idx]

    return most_popular_service, user_test_services.tolist()

def get_recommendations_popular(used_services, popular_services, k=5):
    recommendations = [service for service in popular_services
                      if service not in used_services]
    return recommendations[:k]

def get_recommendations_random(used_services, mid_unique, k):
    available_services = [s for s in range(len(mid_unique)) if s not in used_services]
    return np.random.choice(available_services, size=k, replace=False).tolist()



# funciton for evaluating random and popular
def evaluate_model(X_train, X_test, knn_preds, popular_services, mid_unique, owner_unique, k_values=[1, 3, 5]):
    results = {
        'knn': {k: {'accuracy': [], 'precision': [], 'ndcg': []} for k in k_values},
        'popular': {k: {'accuracy': [], 'precision': [], 'ndcg': []} for k in k_values},
        'random': {k: {'accuracy': [], 'precision': [], 'ndcg': []} for k in k_values}
    }

    for owner_idx, owner_id in enumerate(owner_unique):
        used_services = get_used_services(owner_idx, X_train)
        actual_popular, user_test = get_most_popular_service_test(owner_idx, X_test)
        print("owner_id= ", owner_id, actual_popular, "user_test", user_test)
        if actual_popular is None:
            continue

        for k in k_values:
            # KNN рекомендации
            # Популярные рекомендации
            recs_popular = get_recommendations_popular(used_services, popular_services, k)
            # Рандомные рекомендации
            recs_random = get_recommendations_random(used_services, mid_unique, k)

            # Метрики для всех методов
            for method, recs in [('popular', recs_popular), ('random', recs_random)]:
                y_true = [1 if actual_popular == i else 0 for i in range(len(mid_unique))]
                y_pred = [1 if i in recs else 0 for i in range(len(mid_unique))]
                y_scores = [knn_preds[owner_idx][i] if i in recs else 0 for i in range(len(mid_unique))] if method == 'knn' else [1 if i in recs else 0 for i in range(len(mid_unique))]

                # Accuracy
                accuracy = 1 if actual_popular in recs else 0
                results[method][k]['accuracy'].append(accuracy)

                # Precision
                precision = precision_score(y_true, y_pred)
                results[method][k]['precision'].append(precision)

                # NDCG
                if recs:
                    true_relevance = [1 if i == actual_popular else 0 for i in recs]
                    pred_relevance = [knn_preds[owner_idx][i] for i in recs] if method == 'knn' else [1 if i in recs else 0 for i in recs]
                    ndcg = ndcg_score([true_relevance], [pred_relevance])
                    results[method][k]['ndcg'].append(ndcg)

    # Средние значения метрик
    for method in results:
        for k in k_values:
            results[method][k]['accuracy'] = np.mean(results[method][k]['accuracy'])
            results[method][k]['precision'] = np.mean(results[method][k]['precision'])
            results[method][k]['ndcg'] = np.mean(results[method][k]['ndcg']) if results[method][k]['ndcg'] else 0

    return results

# # # Оценка
k_values = [3, 5, 10, 15]
metrics = evaluate_model(X_train, X_test, preds, popular_services, mid_unique, owner_unique, k_values)

# Вывод результатов
for method in ['knn', 'popular', 'random']:
    print(f"\nResults for {method.upper()} method:")
    for k in k_values:
        print(f"Metrics for k={k}:")
        print(f"Accuracy: {metrics[method][k]['accuracy']:.4f}")
        print(f"Precision: {metrics[method][k]['precision']:.4f}")
        print(f"NDCG: {metrics[method][k]['ndcg']:.4f}")
        print()



# # Вычисление и вывод метрик
# k = [3, 5, 10, 15]
# for el in k:
#     metrics = calculate_metrics(X_test, preds, popular_services, k=el)
#     print(f"Metrics at k={el}")
#     print(f"Accuracy@k: {metrics['accuracy@k']:.4f}")  # новый вывод
#     print(f"Precision@k: {metrics['precision@k']:.4f}")
#     print(f"Recall@k: {metrics['recall@k']:.4f}")
#     print(f"NDCG@k: {metrics['ndcg@k']:.4f}")


# Генерация рекомендаций для JSON
# recommendations = {}
# for i, owner_id in enumerate(owner_unique):
#         used = get_used_services(i, X_train)
#         recs = get_recommendations(i, preds, used, popular_services, k=15)
#         # print(i, owner_id,"used = ",  used, "recs = ", recs)
#         recommendations[owner_id] = [int(mid_unique[idx]) for idx in recs]
#
# # Сохранение в JSON
# with open('recommendations_new.json', 'w', encoding='utf-8') as f:
#     json.dump({"prediction": recommendations}, f, ensure_ascii=False, indent=4)