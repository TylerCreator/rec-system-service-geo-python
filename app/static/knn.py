import pandas as pd
import numpy as np
from numpy import random

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator

import sys
import json
import time

start = time.time()
# считываем из базы данных вызовы сервисов
df_calls = pd.read_csv('app/static/calls.csv', sep=';')
#df_calls = pd.read_csv('CALLS_21_05.csv', sep=';')


# удаляем ненужные колонки (оставялем id, mid, owner, start_time)
# df_calls = df_calls.drop(['classname', 'created_by', 'created_on', 'edited_by', 'edited_on' ], axis = 1)
# df_calls = df_calls.drop(['os_pid','status','is_deleted','updatedAt','createdAt','end_time'], axis=1)

# тут вычислем матрицу частот нормализованную
def prepare_df_old(df, mid_unique, owner_unique):
    X_new = np.zeros((owner_unique.shape[0], mid_unique.shape[0]))
    for i in range(len(owner_unique)):
        for j in range(len(mid_unique)):
            owner = owner_unique[i]
            mid = mid_unique[j]
            X_new[i][j] = df_calls[(df_calls['owner'] == owner) & (df_calls['mid'] == mid)].shape[0]
        cur_sum = sum(X_new[i])
        if cur_sum > 0:
            X_new[i] /= cur_sum
    return X_new

def prepare_df(df, mid_unique, owner_unique):
  #df2.pivot_table(values='X', index=['Y','Z'], columns='X', aggfunc='count')
  pivot = df.pivot_table(values='id', index='owner', columns='mid', aggfunc='count').fillna(0)
  #pivot = df.pivot_table(index='owner', columns='mid', aggfunc='count').fillna(0)
  res = np.zeros((owner_unique.shape[0], mid_unique.shape[0]))
  for i in range(len(owner_unique)):
    for j in range(len(mid_unique)):
      res[i][j] = pivot.loc[owner_unique[i], mid_unique[j]]
    s = np.sum(res[i])
    if s > 0:
      res[i] /= np.sum(res[i])
  return res

# вычисляем косинусное расстояние
def mymetric(A, B):
    res = []
    for i in range(A.shape[0]):
        a = [A[i]]
        b = [B[i]]
        res.append(cosine_similarity(a, b))
    return np.mean(res)

# def get_SVD_preds(X):
#   U, Sigma, Vt = svds(X, k=5)
#   Sigma_diag = np.diag(Sigma)
#   predicted_ratings = np.dot(np.dot(U, Sigma_diag), Vt)
#   return predicted_ratings

class MyNN(BaseEstimator):
    def __init__(self, n_neighbors=3, metric = 'minkowski'):
        self.n_neighbors = n_neighbors + 1
        self.is_fitted_ = False
        self.metric = metric
    def fit(self, X):
        self.X = X
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(X)
        self.distances, self.indices = self.nbrs.kneighbors(X)
        self.is_fitted_ = True
        return self
    def predict(self, X):
        eps = 1e-8
        preds = np.zeros(X.shape)
        #цикл по всем пользователям
        for i in range(self.indices.shape[0]):
            temp = np.zeros(X[0].shape)
            counts = np.zeros(X[0].shape)

            # тут среднее значение высчитывается только по не нулевым значениям
            # for neighbor in self.indices[i][1:]:
            #   temp += X[neighbor]
            #   counts += (X[neighbor] > eps)
            # where_zeroes = (counts < eps)
            # counts[where_zeroes] = 1
            # preds[i] = temp/counts


            for neighbor in self.indices[i][1:]:
                temp += X[neighbor]
            temp /= self.n_neighbors - 1
            preds[i] = temp
        return preds
#df_calls = df_calls.sort_values('start_time')
#df_train = df_calls[:int(df_calls.shape[0] * 0.7)]
#df_test = df_calls[int(df_calls.shape[0] * 0.7):]
mid_unique = df_calls['mid'].unique()
owner_unique = df_calls['owner'].unique()

#X_train = prepare_df(df_train, mid_unique, owner_unique)
#X_test = prepare_df(df_test, mid_unique, owner_unique)

X = prepare_df(df_calls, mid_unique, owner_unique)
#Находим соседей на обучающем множестве, считаем вызовы на тестовом
# тут перебор гипер параметра
# for i in range(1, 9):
#   mnn = MyNN(n_neighbors=i).fit(X_train)
#   preds = mnn.predict(X_train)
#   print(i, mymetric(X_train, preds))

mnn = MyNN(n_neighbors=4, metric='cosine').fit(X)
preds = mnn.predict(X)
#preds = get_SVD_preds(X)

# сортируем сервисы по их популярности
def get_popular_services(df, owner_unique, mid_unique):
    eps = 1e-10
    X_popular_df = X
    X_popular_vector = np.mean(X_popular_df, axis=0)
    zero_pop = np.where(np.abs(X_popular_vector) < eps)
    sort_index = np.argsort(X_popular_vector)[::-1]
    X_popular_with_no_zeros = sort_index[~np.in1d(sort_index, zero_pop)]
    #X_popular_with_no_zeros = np.setdiff1d(sort_index, zero_pop)
    return X_popular_with_no_zeros

# возвращаем какие сервисы пользователь уже использовал
def get_used_services(owner_id, X):
    eps = 1e-10
    n_user = np.where(owner_unique == owner_id)[0][0]
    user_calls = X[n_user]
    return np.where(np.abs(user_calls) > eps)

def gen_similar_for_user(owner_id, preds, used_services, popular_services, n=10):
    eps = 1e-10
    n_user = np.where(owner_unique == owner_id)[0][0]
    user_preds = preds[n_user]
    sorted_services = np.argsort(user_preds)[::-1]
    #print("predicted: ", sorted_services)
    sorted_without_zeros = sorted_services[~np.in1d(sorted_services, np.where(np.abs(user_preds) < eps))]
    sorted_without_used = sorted_without_zeros[~np.in1d(sorted_without_zeros, used_services)]
    #print("predicted without used: ", sorted_without_used)
    popular_services = popular_services[~np.in1d(popular_services, sorted_without_used)]
    popular_services = popular_services[~np.in1d(popular_services, used_services)]
    #print("popular after cleaning: ", popular_services)
    recs = np.append(sorted_without_used, popular_services)[:n]
    return recs

def get_mid_names(indices, mid_unique):
    res = []
    for i in indices:
        res.append(mid_unique[i])
    return np.array(res)

popular_services = get_popular_services(df_calls, owner_unique, mid_unique)
X = prepare_df(df_calls, mid_unique, owner_unique)
def get_answer(owner_id, preds):
    used_services = get_used_services(owner_id, X)
    recs = gen_similar_for_user(owner_id, preds, used_services, popular_services, n=15)
    return get_mid_names(recs, mid_unique)

answer = {}
for owner_id in owner_unique:
    if ('cookies' not in owner_id):
        answer[owner_id] = get_answer(owner_id, preds).tolist()
with open('app/static/recomendations_knn.json', 'w', encoding='utf-8') as f:
    json.dump({"prediction": answer }, f, ensure_ascii=False, indent=4)
print(json.dumps({"prediction": answer }))

sys.stdout.flush()