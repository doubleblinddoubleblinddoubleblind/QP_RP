import pickle

from tree_sitter import Language, Parser
from paths import *

strEL=' _EL_ '
batchSizeForSaveASTs=500


dictDatasetInfos={}
dictDatasetInfos['AdvTest']={'templateFileName':'Hello.py','lstSplits':['codebase.jsonl','test.jsonl','valid.jsonl','train.jsonl'],'nameDS':'AdvTest','queryFile':'test__jsonl.test__jsonl','trainFile':'train__jsonl.train__jsonl','topType':'function_definition'}
dictDatasetInfos['cosqa']={'templateFileName':'Hello.py','lstSplits':['code_idx_map.txt','test.json','valid.json','train.json'],'nameDS':'cosqa','queryFile':'test__json.code_idx_map__txt','trainFile':'train__json.train__json','topType':'function_definition'}
dictDatasetInfos['java']={'templateFileName':'Hello.java','lstSplits':['codebase.jsonl','test.jsonl','valid.jsonl','train.jsonl'],'nameDS':'csnOrg','queryFile':'test','trainFile':'train','topType':'method_declaration'}
dictDatasetInfos['python']={'templateFileName':'Hello.py','lstSplits':['codebase.jsonl','test.jsonl','valid.jsonl','train.jsonl'],'nameDS':'csnOrg-python','queryFile':'test','trainFile':'train','topType':'function_definition'}
dictDatasetInfos['javascript']={'templateFileName':'Hello.js','lstSplits':['codebase.jsonl','test.jsonl','valid.jsonl','train.jsonl'],'nameDS':'csnOrg-javascript','queryFile':'test','trainFile':'train','topType':'function_declaration'}
dictDatasetInfos['php']={'templateFileName':'Hello.php','lstSplits':['codebase.jsonl','test.jsonl','valid.jsonl','train.jsonl'],'nameDS':'csnOrg-php','queryFile':'test','trainFile':'train','topType':'function_definition'}
dictDatasetInfos['go']={'templateFileName':'Hello.go','lstSplits':['codebase.jsonl','test.jsonl','valid.jsonl','train.jsonl'],'nameDS':'csnOrg-go','queryFile':'test','trainFile':'train','topType':'method_declaration'}
dictDatasetInfos['ruby']={'templateFileName':'Hello.rb','lstSplits':['codebase.jsonl','test.jsonl','valid.jsonl','train.jsonl'],'nameDS':'csnOrg-ruby','queryFile':'test','trainFile':'train','topType':'method'}

# This dictionary is the tuned weights for merging the embeddings between original model and NLPT-AST models (rq1-2) and between original-NLPT-AST for the combined models.
dictTuneWeights=pickle.load(open('merging_weights.pkl','rb'))