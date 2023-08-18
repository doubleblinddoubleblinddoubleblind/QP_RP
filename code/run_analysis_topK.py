import pickle

import pandas as pd

from utils import *
import time
from configParams import *
# from UtilGetReduction import *
# from UtilWeightCalculator import *

lstProjectNames=['AdvTest','cosqa','java','python','javascript','go','php','ruby']
lstModels=['baseline-unixcoder','nlpt','ast','combine','ast-optimal']
cacheSize=500
strTemplateName='templateCode'

fopSearchResults=fopProjectRoot+ 'results_v1/'
lstTopKSelect=list(range(1,6))

for idxModel in range(0,len(lstModels)):
    currentModel=lstModels[idxModel]
    dictListTopK={}
    for idxProject in range(0,len(lstProjectNames)):
        currentDataset=lstProjectNames[idxProject]
        dictPerProjectInfo=dictDatasetInfos[currentDataset]
        lstSplits=dictPerProjectInfo['lstSplits']
        lstSplits=[lstSplits[0],lstSplits[1]]
        currentTemplateFileName=dictPerProjectInfo['templateFileName']
        currentTemplateExtension=currentTemplateFileName.split('.')[1]
        fopDetailFolderPerProject=fopSearchResults+'{}/{}/'.format(currentModel,currentDataset)
        fpPerProjectRankDetails=fopDetailFolderPerProject+'pred/details_rank/{}__unixcoder__768.csv'.format(currentDataset)
        fpPerProjectPredictDetails = fopDetailFolderPerProject + 'pred/predicted_rank/{}__unixcoder__768.csv'.format(
            currentDataset)
        dfPPRankDetails=pd.read_csv(fpPerProjectRankDetails)
        dfPPPredictedDetails = pd.read_csv(fpPerProjectPredictDetails)

        dictRank={}
        for k in lstTopKSelect:
            dictRank[k]=0
        lenOfEntities=len(dfPPRankDetails)

        for idxCsv in range(0,len(dfPPRankDetails)):
            try:
                keyUrl=str(dfPPRankDetails['QueryId'][idxCsv])
                val=dfPPRankDetails['Rank'][idxCsv]
                for k in lstTopKSelect:
                    if val<=k:
                        dictRank[k]+=1
            except Exception as e:
                traceback.print_exc()

        for idxCsv in range(0,len(dfPPPredictedDetails)):
            try:
                keyUrl=str(dfPPPredictedDetails['QueryId'][idxCsv])
                val=dfPPPredictedDetails['PredictTop1Id'][idxCsv]
            except Exception as e:
                traceback.print_exc()

        strContentLine='{}\t{}\t'.format(currentModel,currentDataset)+'\t'.join(['{}'.format(dictRank[k]/lenOfEntities) for k in lstTopKSelect])
        for k in lstTopKSelect:
            if k not in dictListTopK.keys():
                dictListTopK[k]=[]
            dictListTopK[k].append(dictRank[k]/lenOfEntities)
        # print(strContentLine)
    print('{}\tAverage\t'.format(currentModel)+'\t'.join(['{}'.format(mean(dictListTopK[k])) for k in lstTopKSelect]))





