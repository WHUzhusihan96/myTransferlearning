# code by chenchiwei
# -*- coding: UTF-8 -*- 
import numpy as np
from sklearn import tree
import scipy.io
import scipy.linalg
import sklearn.metrics

# H 测试样本分类结果
# TrainS 原训练样本 np数组
# TrainA 辅助训练样本
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数

def tradaboost(trans_S, trans_A, label_S, label_A, test, N):

    # 文章中的T->训练数据
    trans_data = np.concatenate((trans_A, trans_S), axis=0)# 所有训练数据拼接在一起,按列
    trans_label = np.concatenate((label_A, label_S), axis=0)# 训练数据对应的标签

    row_A = trans_A.shape[0]# 获得不同分布的样本量
    row_S = trans_S.shape[0]# 获得同分布的样本量
    row_T = test.shape[0]# 测试样本量

    test_data = np.concatenate((trans_data, test), axis=0)

    # 初始化权重
    weights_A = np.ones([row_A, 1]) / row_A# 列向量权重
    weights_S = np.ones([row_S, 1]) / row_S# 列向量权重
    weights = np.concatenate((weights_A, weights_S), axis=0)# 竖着拼接起来

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))# 根据文章种的算式来的

    # 存储每次迭代的标签和bata值？
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    predict = np.zeros([row_T])# 为预测值建立一个数组

    print('params initial finished.')
    trans_data = np.asarray(trans_data, order='C')# asarray将数组cut过去，随着数组改变而改变
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        P = calculate_P(weights, trans_label)# p为权重矩阵的归一化后的新矩阵，weights与原文wt对应

        result_label[:, i] = train_classify(trans_data, trans_label, test_data, P)# 将第i次预测的标签存储在第i列上。

        # print('result,', result_label[:, i], row_A, row_S, i, result_label.shape)

        error_rate = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
                                          weights[row_A:row_A + row_S, :])
        # print('Error rate:', error_rate)
        if error_rate > 0.5:
            error_rate = 0.5
        if error_rate == 0:
            N = i
            break  # 防止过拟合
            # error_rate = 0.001

        bata_T[0, i] = error_rate / (1 - error_rate)# 根据原文的更新公式更新Beta_T

        # 根据原文的调整公式，更新样本权重
        # 调整源域样本权重
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                                               (-np.abs(result_label[row_A + j, i] - label_S[j])))

        # 调整辅域样本权重
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))
    # print bata_T
    for i in range(row_T):
        # 跳过训练数据的标签
        # 根据原文计算左右两边，通过比较大小给出类别
        left = np.sum(
            result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
            # print left, right, predict[i]

    return predict


def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classify(trans_data, trans_label, test_data, P):
    clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)


def calculate_error_rate(label_R, label_H, weight):#分别是真实标签，预测标签和权重值。
    total = np.sum(weight)

    # print(weight[:, 0] / total)
    # print(np.abs(label_R - label_H))
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))#与原文的error计算对应
