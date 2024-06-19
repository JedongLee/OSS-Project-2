import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 파일 경로

ratings_path = 'ratings.dat'

# 데이터프레임으로 변환
ratings = pd.read_csv(ratings_path, sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding='latin-1')

# 피벗 테이블 생성
ratings_pivot = ratings.pivot(index='UserID', columns='MovieID', values='Rating')
#print((ratings_pivot==0).sum().sum())


# 결측값을 0으로 대체
ratings_pivot.fillna(0, inplace=True)

# KMeans 클러스터링 적용
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(ratings_pivot)

cluster1=ratings_pivot[clusters==0]
cluster2=ratings_pivot[clusters==1]
cluster3=ratings_pivot[clusters==2]

def CR(rating):
    column = rating.shape[1]
    matrix = pd.DataFrame(np.zeros((column, column)), index=rating.columns, columns=rating.columns)

    for i in range(column):
        for j in range(i + 1, column):
            temp = (rating.iloc[:, i] < rating.iloc[:, j]).sum() - (rating.iloc[:, i] > rating.iloc[:, j]).sum()
            matrix.iloc[i, j] = temp
            matrix.iloc[j, i] = -temp

    matrix = matrix.applymap(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
    result = matrix.sum(axis=0).sort_values(ascending=False).index[:10].values
    print(result)
def AU(rating):
    print(rating.sum(axis=0).sort_values(ascending=False).index[:10].values)

def AVG(rating):
    print(rating.mean(axis=0).sort_values(ascending=False).index[:10].values)

def SC(rating):
    print((rating>0).sum(axis=0).sort_values(ascending=False).index[:10].values)

def AV(rating):
    print((rating>4).sum(axis=0).sort_values(ascending=False).index[:10].values)


def BC(rating):
    print(rating[rating!=0].rank(axis=1).sum(axis=0).sort_values(ascending=False).index[:10].values)


# 클러스터링 결과 출력


print("----------------AU------------------")
AU(cluster1)
AU(cluster2)
AU(cluster3)
print()

print("----------------AVG------------------")
AVG(cluster1)
AVG(cluster2)
AVG(cluster3)
print()

print("----------------SC------------------")
SC(cluster1)
SC(cluster2)
SC(cluster3)
print()

print("----------------AV------------------")
AV(cluster1)
AV(cluster2)
AV(cluster3)
print()

print("----------------BC------------------")
BC(cluster1)
BC(cluster2)
BC(cluster3)
print()


print("----------------CR------------------")
CR(cluster1)
CR(cluster2)
CR(cluster3)


