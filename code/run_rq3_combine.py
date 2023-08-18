import pickle

from utils import *
import time
from configParams import *
# from UtilGetReduction import *
# from UtilWeightCalculator import *

fopRootOriginalEmb=fopProjectRoot+'data/vectors/original/'
fopRootNextEmb=fopProjectRoot+'data/vectors/nlpt/'
fopRootSecongEmb=fopProjectRoot+'data/vectors/ast/'


lstProjectNames=['cosqa','AdvTest','java','python','javascript','go','php','ruby']
embeddingModel='unixcoder'
lstSizeOfReduction=[768]
currentReductionType='pca'
fopSearchResults=fopProjectRoot+'results/combine/'
createDirIfNotExist(fopSearchResults)
dictConfigWeightTune=dictTuneWeights['combine']



for currentProjectName in lstProjectNames:
    try:
        combineWeightA = dictConfigWeightTune[currentProjectName][0]
        combineWightB = dictConfigWeightTune[currentProjectName][1]
        fpSummary = fopSearchResults + 'summary_{}.txt'.format(currentProjectName)
        f1 = open(fpSummary, 'w')
        f1.write('Project\tEmbeddingModel\tReductSize\tMRR\tDuration\n')
        f1.close()

        # nameTestPkl='{}.test.pkl'.format(currentProjectName)
        fopOutputOriginalPerProject = '{}/{}/pred/'.format(fopSearchResults, currentProjectName)
        createDirIfNotExist(fopOutputOriginalPerProject)

        fopDetailsRank = fopOutputOriginalPerProject + 'details_rank/'
        fopDetailsTop = fopOutputOriginalPerProject + 'details_top/'
        fopPredictedRank = fopOutputOriginalPerProject + 'predicted_rank/'
        fopStoreScoreOrg = fopOutputOriginalPerProject + 'store_score_org/'
        fopStoreScoreAug = fopOutputOriginalPerProject + 'store_score_aug/'
        fopStoreSortedIndex = fopOutputOriginalPerProject + 'store_sorted_ids/'
        createDirIfNotExist(fopStoreScoreOrg)
        createDirIfNotExist(fopStoreScoreAug)
        createDirIfNotExist(fopStoreSortedIndex)
        createDirIfNotExist(fopDetailsRank)
        createDirIfNotExist(fopDetailsTop)
        createDirIfNotExist(fopPredictedRank)

        dictInfoPerDataset = dictDatasetInfos[currentProjectName]
        lstSplits = dictInfoPerDataset['lstSplits']

        # fopInsideFeat = fopModels + '{}/'.format(currentProjectName)
        # fpQueryExpPkl = fopInsideFeat + 'labelsASTPropsWithURL_{}.pkl'.format(lstSplits[1])
        # fpCandExpPkl = fopInsideFeat + 'labelsASTPropsWithURL_{}.pkl'.format(lstSplits[0])
        # fpQueryPredPkl=fopRootOfMLEstimators+'{}/{}/pred.pkl'.format(currentProjectName,nameOfEstiConfig)
        # fpQueryCandPk=fopModels+'{}_v1/{}/toy-ende/unixcoder.test.emb.100.pkl'.format(currentProjectName,currentDepth)
        # dictQueryAndCands=pickle.load(open(fpQueryCandPk,'rb'))
        # dictQueryEmb=dictQueryAndCands['queries']
        # dictCandEmb=dictQueryAndCands['candidates']
        # dictTemp={}
        # for key in dictQueryEmb.keys():
        #     dictTemp[int(key)]=dictQueryEmb[key]
        # dictQueryEmb=dictTemp
        # dictTemp = {}
        # for key in dictCandEmb.keys():
        #     dictTemp[int(key)] = dictCandEmb[key]
        # dictCandEmb = dictTemp


        # setQueryKeys = set(list(dictQueryEmb.keys()))
        # # print('set {}'.format(setQueryKeys))
        # # input('aaaa')
        # if len(list(dictQueryEmb.values()))>0:
        #     augmentedSize = len(list(dictQueryEmb.values())[0]['exp'])
        # setCandKeys = set(list(dictCandEmb.keys()))
        # print('set {}'.format(setCandKeys))
        # input('aaaa')

        for reductSize in lstSizeOfReduction:

            try:
                fpQueryPkl = '{}/{}.{}.pkl'.format(fopRootOriginalEmb,
                                                      dictInfoPerDataset['nameDS'], 'test')
                strNextNamePkl = 'test__jsonl.codebase__jsonl'
                if currentProjectName == 'AdvTest':
                    strNextNamePkl = 'test__jsonl.test__jsonl'
                fpNextQueryPkl = '{}/{}.{}.pkl'.format(fopRootNextEmb,
                                                      currentProjectName, strNextNamePkl)
                fpSecondQueryPkl = '{}/{}.{}.pkl'.format(fopRootSecongEmb,
                                                          currentProjectName, strNextNamePkl)

                strItemId = '{}__{}__{}'.format(currentProjectName, embeddingModel, reductSize)
                print('begin {}'.format(strItemId))
                dictQueriesCands = pickle.load(open(fpQueryPkl, 'rb'))
                dictAllQueries = dictQueriesCands['queries']
                dictAllCands = dictQueriesCands['candidates']
                # print('key {}'.format(dictAllQueries.keys()))
                dictNewIdQueries = {}
                dictNewIdCands = {}
                for key in dictAllQueries.keys():
                    dictNewIdQueries[key] = dictAllQueries[key]
                for key in dictAllCands.keys():
                    dictNewIdCands[key] = dictAllCands[key]

                dictNewIdQueries = dict(sorted(dictNewIdQueries.items()))
                dictNewIdCands = dict(sorted(dictNewIdCands.items()))
                listKeyQueries = [item for item in list(dictNewIdQueries.keys())]
                listKeyCands = [item for item in list(dictNewIdCands.keys())]
                listVectorQueries = list(dictNewIdQueries.values())
                listVectorCands = list(dictNewIdCands.values())
                listVectorQueries = [a.tolist() for a in listVectorQueries]
                listVectorCands = [a.tolist() for a in listVectorCands]
                # print('len {}'.format(len(listVectorQueries)))
                # input('bbb')

                dictNextQueriesCands = pickle.load(open(fpNextQueryPkl, 'rb'))
                dictNextAllQueries = dictNextQueriesCands['queries']
                dictNextAllCands = dictNextQueriesCands['candidates']
                # print('key {}'.format(dictAllQueries.keys()))
                dictNextNewIdQueries = {}
                dictNextNewIdCands = {}
                for key in dictNextAllQueries.keys():
                    dictNextNewIdQueries[key] = dictNextAllQueries[key]
                for key in dictNextAllCands.keys():
                    dictNextNewIdCands[key] = dictNextAllCands[key]

                dictNextNewIdQueries = dict(sorted(dictNextNewIdQueries.items()))
                dictNextNewIdCands = dict(sorted(dictNextNewIdCands.items()))
                listNextKeyQueries = [item for item in list(dictNextNewIdQueries.keys())]
                listNextKeyCands = [item for item in list(dictNextNewIdCands.keys())]
                listNextVectorQueries = list(dictNextNewIdQueries.values())
                listNextVectorCands = list(dictNextNewIdCands.values())
                listNextVectorQueries = [a.tolist() for a in listNextVectorQueries]
                listNextVectorCands = [a.tolist() for a in listNextVectorCands]

                dictSecondQueriesCands = pickle.load(open(fpSecondQueryPkl, 'rb'))
                dictSecondAllQueries = dictSecondQueriesCands['queries']
                dictSecondAllCands = dictSecondQueriesCands['candidates']
                # print('key {}'.format(dictAllQueries.keys()))
                dictSecondNewIdQueries = {}
                dictSecondNewIdCands = {}
                for key in dictSecondAllQueries.keys():
                    dictSecondNewIdQueries[key] = dictSecondAllQueries[key]
                for key in dictSecondAllCands.keys():
                    dictSecondNewIdCands[key] = dictSecondAllCands[key]

                dictSecondNewIdQueries = dict(sorted(dictSecondNewIdQueries.items()))
                dictSecondNewIdCands = dict(sorted(dictSecondNewIdCands.items()))
                listSecondKeyQueries = [item for item in list(dictSecondNewIdQueries.keys())]
                listSecondKeyCands = [item for item in list(dictSecondNewIdCands.keys())]
                listSecondVectorQueries = list(dictSecondNewIdQueries.values())
                listSecondVectorCands = list(dictSecondNewIdCands.values())
                listSecondVectorQueries = [a.tolist() for a in listSecondVectorQueries]
                listSecondVectorCands = [a.tolist() for a in listSecondVectorCands]

                if len(listVectorQueries[0]) > reductSize:
                    nl_vecs, code_vecs = getReductionEmb(listVectorQueries, listVectorCands, currentReductionType,
                                                         reductSize)
                    nlNext_vecs, codeNext_vecs = getReductionEmb(listNextVectorQueries, listNextVectorCands, currentReductionType,
                                                         reductSize)
                    nlSecond_vecs, codeSecond_vecs = getReductionEmb(listSecondVectorQueries, listSecondVectorCands,
                                                                 currentReductionType,
                                                                 reductSize)
                else:
                    nl_vecs, code_vecs = listVectorQueries, listVectorCands
                    nlNext_vecs, codeNext_vecs = listNextVectorQueries, listNextVectorCands
                    nlSecond_vecs, codeSecond_vecs = listSecondVectorQueries, listSecondVectorCands

                # nlAug_vecs = []
                # codeAug_vecs = []
                # numExistInQuery=0
                # for indexKey in range(0, len(listKeyQueries)):
                #     keyQuery = listKeyQueries[indexKey]
                #     if keyQuery in dictQueryEmb.keys():
                #         valAug = dictQueryEmb[keyQuery]
                #         lstClonePred = copy.copy(valAug['pred'])
                #         lstAugPred = [(it) for it in lstClonePred]
                #         nlAug_vecs.append(lstAugPred)
                #         numExistInQuery+=1
                #         # codeAug_vecs.append(lstAugExp)
                #         # print('key found {}'.format(keyQuery))
                #     else:
                #         lstAugExp = lstAugPred = [0.00001 for i in range(0, augmentedSize)]
                #         nlAug_vecs.append(lstAugPred)
                #         # codeAug_vecs.append(lstAugExp)
                #         # print('go here {}'.format(augmentedSize))
                # numExistInCand=0
                # for indexKey in range(0, len(listKeyCands)):
                #     keyCand = listKeyCands[indexKey]
                #     if keyCand in dictCandEmb.keys():
                #         valAug = dictCandEmb[keyCand]
                #         lstCloneExp = copy.copy(valAug['exp'])
                #         lstAugExp = [(it) for it in lstCloneExp]
                #         codeAug_vecs.append(lstAugExp)
                #         numExistInCand+=1
                #         # codeAug_vecs.append(lstAugExp)
                #         # print('key found {}'.format(keyQuery))
                #     else:
                #         lstAugExp = lstAugPred = [0.00001 for i in range(0, augmentedSize)]
                #         codeAug_vecs.append(lstAugPred)
                #         # codeAug_vecs.append(lstAugExp)
                #         # print('go here {}'.format(augmentedSize))
                # print('query percent {} {} - cand percent {} {}'.format(numExistInQuery,numExistInQuery/len(listKeyQueries),numExistInCand,numExistInCand/len((listKeyCands))))

                nl_vecs = [[a] for a in nl_vecs]
                code_vecs = [[a] for a in code_vecs]
                code_vecs = np.concatenate(code_vecs, 0)
                nl_vecs = np.concatenate(nl_vecs, 0)

                nlNext_vecs = [[a] for a in nlNext_vecs]
                codeNext_vecs = [[a] for a in codeNext_vecs]
                codeNext_vecs = np.concatenate(codeNext_vecs, 0)
                nlNext_vecs = np.concatenate(nlNext_vecs, 0)

                nlSecond_vecs = [[a] for a in nlSecond_vecs]
                codeSecond_vecs = [[a] for a in codeSecond_vecs]
                codeSecond_vecs = np.concatenate(codeSecond_vecs, 0)
                nlSecond_vecs = np.concatenate(nlSecond_vecs, 0)

                # nlAug_vecs = [[a] for a in nlAug_vecs]
                # codeAug_vecs = [[a] for a in codeAug_vecs]
                # codeAug_vecs = np.concatenate(codeAug_vecs, 0)
                # nlAug_vecs = np.concatenate(nlAug_vecs, 0)

                # print('{} {} {}'.format(len(listVectorCands),type(listVectorCands[0]),listVectorCands[0]))
                start_time = time.time()
                scores = csm(nl_vecs, code_vecs)
                scores = adjustScoreForMatrix(scores)
                scoresAug = csm(nlNext_vecs, codeNext_vecs)
                scoresAug = adjustScoreForMatrix(scoresAug)
                scoresSec = csm(nlSecond_vecs, codeSecond_vecs)
                scoresSec = adjustScoreForMatrix(scoresSec)
                scores = scores * (1 - combineWeightA) + scoresAug * combineWeightA+scoresSec*combineWightB
                sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
                del scores,scoresAug

                ranks = []
                realRanks = []
                lstPredictedTop1s = []
                # dictDetailResults={}
                # zipData = zip(listKeyQueries, sort_ids)
                currentIndexSortedIndex = -1
                dictBatchIndexSorted = {}
                # for url, sort_id in zipData:
                for indexScore in range(0, len(listKeyQueries)):
                    url = listKeyQueries[indexScore]
                    # newIndexSortedIndex=indexScore//cacheSizeForVector
                    # if newIndexSortedIndex!=currentIndexSortedIndex:
                    #     currentIndexSortedIndex = newIndexSortedIndex
                    #     fpBatchSortedIndex = fopStoreSortedIndex + '{}_{}_{}.pkl'.format(reductSize, embeddingModel,
                    #                                                                      currentIndexSortedIndex)
                    #     dictBatchIndexSorted=pickle.load(open(fpBatchSortedIndex,'rb'))

                    # sort_id=dictBatchIndexSorted[indexScore]
                    sort_id = sort_ids[indexScore]
                    rank = 0
                    find = False
                    # lstTop1000PerKey=[]
                    lstPredictedTop1s.append(listKeyCands[sort_id[0]])
                    for idx in sort_id[:1000]:
                        # lstTop1000PerKey.append(listKeyCands[idx])
                        if find is False:
                            rank += 1
                        if listKeyCands[idx] == url:
                            find = True
                    # dictDetailResults[url]=lstTop1000PerKey
                    if find:
                        realRanks.append(rank)
                        ranks.append(1 / rank)

                    else:
                        ranks.append(0)
                        realRanks.append(1001)
                del sort_id, find, rank

                mrrScore = np.mean(ranks)
                duration = (time.time() - start_time)
                lstKeyAndRank = ['{},{}'.format(listKeyQueries[i], realRanks[i]) for i in
                                 range(0, len(listKeyQueries))]
                lstKeyAndRank = ['QueryId,Rank'] + lstKeyAndRank
                # strItemId = '{}__augOnly'.format(currentProjectName)
                f1 = open(fopDetailsRank + strItemId + '.csv', 'w')
                f1.write('\n'.join(lstKeyAndRank))
                f1.close()

                lstPredictAndRank = ['{},{}'.format(listKeyQueries[i], lstPredictedTop1s[i]) for i in
                                     range(0, len(listKeyQueries))]
                lstPredictAndRank = ['QueryId,PredictTop1Id'] + lstPredictAndRank
                f1 = open(fopPredictedRank + strItemId + '.csv', 'w')
                f1.write('\n'.join(lstPredictAndRank))
                f1.close()
                # fpDetailRanksAndTops= fopDetailsTop + strItemId + '.csv'
                # exportDictToExcel(fpDetailRanksAndTops, realRanks, dictDetailResults)

                strLineMRRScore = '{}\t{}\t{}\t{}'.format(strItemId, currentProjectName,
                                                          mrrScore, duration)
                print(strLineMRRScore)
                f1 = open(fpSummary, 'a')
                f1.write(strLineMRRScore + '\n')
                f1.close()
                ranks = None
                realRanks = None
                nl_vecs = None
                code_vecs = None
                zipData = None
                # listKeyQueries=None
                scores = None
                sort_ids = None

            except Exception as e:
                traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
