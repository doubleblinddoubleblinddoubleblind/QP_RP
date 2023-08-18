import sys, os, traceback
import ast
from statistics import mean

import torch
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
from sklearn.decomposition import PCA
import json
# from sklearn.manifold import TSNE
from torchmetrics import MeanAbsolutePercentageError
def csm(A,B):
    num=np.dot(A,B.T)/(np.sqrt(np.sum(A**2,axis=1)[:,np.newaxis])*np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:])
    return num
def adjustScoreForMatrix(scores):
    try:
        minValue=np.min(scores)
        maxValue=np.max(scores)
        distance=maxValue-minValue
        scores=(scores-minValue)/distance
    except Exception as e:
        traceback.print_exc()
    return scores

def getReductionEmb(nl_vecs,code_vecs,reductType,reductSize):
    lenNLVecs=len(nl_vecs)
    lenCodeVecs=len(code_vecs)
    nl_vecs_transform=[]
    code_vecs_transform=[]
    all_vecs=nl_vecs+code_vecs
    # print('len all {}'.format(len(all_vecs)))
    if reductType=='adhoc':
        all_vecs_transform=[element[:reductSize] for element in all_vecs]
        nl_vecs_transform = all_vecs_transform[:lenNLVecs]
        code_vecs_transform = all_vecs_transform[lenNLVecs:]
    elif reductType=='pca':
        pca = PCA(n_components=reductSize)
        all_vecs_transform =pca.fit_transform(all_vecs)
        nl_vecs_transform = all_vecs_transform[:lenNLVecs]
        code_vecs_transform = all_vecs_transform[lenNLVecs:]
    # else:
    #     tsne = TSNE(n_components=reductSize,
    #                 perplexity=40,
    #                 random_state=42,
    #                 n_iter=5000,
    #                 n_jobs=-1)
    #     all_vecs_transform = tsne.fit_transform(all_vecs)
    #     nl_vecs_transform = all_vecs_transform[:lenNLVecs]
    #     code_vecs_transform = all_vecs_transform[lenNLVecs:]

    return nl_vecs_transform.tolist(),code_vecs_transform.tolist()


def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.makedirs(fopOutput, exist_ok=True)
        #print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")
