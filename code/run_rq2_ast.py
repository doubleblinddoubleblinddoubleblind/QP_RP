import pickle

from utils import *
import time
from configParams import *
# from UtilGetReduction import *
# from UtilWeightCalculator import *

fopRootOriginalEmb=fopProjectRoot+'data/vectors/original/'
fopRootNextEmb=fopProjectRoot+'data/vectors/ast/'

lstProjectNames=['AdvTest','cosqa','java','python','javascript','go','php','ruby']
lstSizeOfReduction=[768]
currentReductionType='pca'
embeddingModel='unixcoder'
dictConfigWeightTune=dictTuneWeights['ast']


currentInputFeatureSize=768
fopGenerateExpectedVectors=fopProjectRoot+'vectors/'
fopSearchResults=fopProjectRoot+'results/ast/'
createDirIfNotExist(fopSearchResults)


for currentProjectName in lstProjectNames:
    try:
        combineWeight = dictConfigWeightTune[currentProjectName][0]
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

        for reductSize in lstSizeOfReduction:

            try:
                fpQueryPkl = '{}/{}.{}.pkl'.format(fopRootOriginalEmb,
                                                      dictInfoPerDataset['nameDS'], 'test')
                strNextNamePkl='test__jsonl.codebase__jsonl'
                if currentProjectName=='AdvTest':
                    strNextNamePkl='test__jsonl.test__jsonl'
                fpNextQueryPkl = '{}/{}.{}.pkl'.format(fopRootNextEmb,
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

                if len(listVectorQueries[0]) > reductSize:
                    nl_vecs, code_vecs = getReductionEmb(listVectorQueries, listVectorCands, currentReductionType,
                                                         reductSize)
                    nlNext_vecs, codeNext_vecs = getReductionEmb(listNextVectorQueries, listNextVectorCands, currentReductionType,
                                                         reductSize)
                else:
                    nl_vecs, code_vecs = listVectorQueries, listVectorCands
                    nlNext_vecs, codeNext_vecs = listNextVectorQueries, listNextVectorCands


                nl_vecs = [[a] for a in nl_vecs]
                code_vecs = [[a] for a in code_vecs]
                code_vecs = np.concatenate(code_vecs, 0)
                nl_vecs = np.concatenate(nl_vecs, 0)

                nlNext_vecs = [[a] for a in nlNext_vecs]
                codeNext_vecs = [[a] for a in codeNext_vecs]
                codeNext_vecs = np.concatenate(codeNext_vecs, 0)
                nlNext_vecs = np.concatenate(nlNext_vecs, 0)

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
                scores = scores * (1 - combineWeight) + scoresAug * combineWeight
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
