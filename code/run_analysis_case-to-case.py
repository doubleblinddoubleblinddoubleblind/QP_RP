import pickle

import pandas as pd

from utils import *
import time
from configParams import *

def get2DictFrom2Csvs(fpPerProjectRankDetails,fpPerProjectPredictDetails):
    dfPPRankDetails = pd.read_csv(fpPerProjectRankDetails)
    dfPPPredictedDetails = pd.read_csv(fpPerProjectPredictDetails)
    lenOfEntities = len(dfPPRankDetails)
    dictDetailRank = {}
    for idxCsv in range(0, len(dfPPRankDetails)):
        try:
            keyUrl = str(dfPPRankDetails['QueryId'][idxCsv])
            val = dfPPRankDetails['Rank'][idxCsv]
            # for k in lstTopKSelect:
            #     if val<=k:
            #         dictRank[k]+=1
            dictDetailRank[keyUrl] = val
        except Exception as e:
            traceback.print_exc()
    dictPredictResult = {}
    for idxCsv in range(0, len(dfPPPredictedDetails)):
        try:
            keyUrl = str(dfPPPredictedDetails['QueryId'][idxCsv])
            val = str(dfPPPredictedDetails['PredictTop1Id'][idxCsv])
            dictPredictResult[keyUrl] = val
        except Exception as e:
            traceback.print_exc()
    return dictDetailRank,dictPredictResult,lenOfEntities


lstProjectNames=['AdvTest','cosqa','java','python','javascript','go','php','ruby']
lstModels=['nlpt','ast','combine','ast-optimal']
lstModels=['combine']
originalModel='baseline-unixcoder'
cacheSize=500
strTemplateName='templateCode'
fopSearchResults=fopProjectRoot+ 'results_v1/'
fopResultAnalyze=fopProjectRoot+ 'analyze/'
createDirIfNotExist(fopResultAnalyze)

for idxModel in range(0,len(lstModels)):
    currentModel=lstModels[idxModel]
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
        dictModelDetailRank, dictModelPredictResult, lenModelOfEntities=get2DictFrom2Csvs(fpPerProjectRankDetails,fpPerProjectPredictDetails)
        fopOrgDetailFolderPerProject = fopSearchResults + '{}/{}/'.format(originalModel, currentDataset)
        fpOrgPerProjectRankDetails = fopOrgDetailFolderPerProject + 'pred/details_rank/{}__unixcoder__768.csv'.format(
            currentDataset)
        fpOrgPerProjectPredictDetails = fopOrgDetailFolderPerProject + 'pred/predicted_rank/{}__unixcoder__768.csv'.format(
            currentDataset)
        dictOrgModelDetailRank, dictOrgModelPredictResult, lenOrgModelOfEntities = get2DictFrom2Csvs(fpOrgPerProjectRankDetails,
                                                                                            fpOrgPerProjectPredictDetails)
        lstAllKey=list(dictOrgModelDetailRank.keys())
        numBetter=0
        numDraw=0
        numWorse=0
        for key in lstAllKey:
            # print(key)
            if key in dictModelDetailRank.keys():
                valPred=dictModelDetailRank[key]
                valOrg=dictOrgModelDetailRank[key]
                # print(valOrg)
                if valPred<valOrg:
                    numBetter+=1
                elif valPred>valOrg:
                    numWorse+=1
                else:
                    numDraw+=1
        strContentLine='{}\t{}\t{}\t{}\t{}'.format(currentModel,currentDataset,(numBetter/lenModelOfEntities),numDraw/lenModelOfEntities,numWorse/lenModelOfEntities)
        print(strContentLine)




