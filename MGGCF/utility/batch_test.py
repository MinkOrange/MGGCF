import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data_v1 import *
import heapq
import math
import numpy as np

args = parse_args()
K = args.K

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size,num_negatives=args.num_negatives)

def getHitRatio( ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

def group_test(sess, model, users_to_test, drop_flag=False):
    result = {'ndcg':0.,'hit_ratio': 0.}
    hits, ndcgs = [],[]
    for u in users_to_test:
        for test_item in data_generator.group_test_set[u]:
            test_items = data_generator.group_neg_pools[(u,test_item)]
            test_items.append(test_item)
            if drop_flag == False:
                rate = sess.run(model.g_batch_ratings, {model.groups: [u], model.g_pos_items:test_items })
            else:
                rate = sess.run(model.g_batch_ratings, {model.groups: [u], model.g_pos_items: test_items,
                                                        model.mess_dropout: [0.] * len(eval(args.layer_size)),
                                                        model.node_dropout: [0.]*len(eval(args.layer_size))})
            map_item_score ={}
            for i in range(len(test_items)):
                map_item_score[test_items[i]] = rate[0][i]
            ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
            hit = getHitRatio(ranklist,test_item)
            ndcg = getNDCG(ranklist,test_item)
            hits.append(hit)
            ndcgs.append(ndcg)
    result['ndcg'] = np.array(ndcgs).mean()
    result['hit_ratio'] = np.array(hits).mean()

    return result
def user_test(sess, model, users_to_test, drop_flag=False):
    result = {'ndcg': 0., 'hit_ratio': 0.}
    hits, ndcgs = [], []
    for u in users_to_test:
        for test_item in data_generator.user_test_set[u]:
            test_items = data_generator.user_neg_pools[(u,test_item)]
            test_items.append(test_item)
            if drop_flag == False:
                rate = sess.run(model.u_batch_ratings, {model.users: [u], model.u_pos_items: test_items})
            else:
                rate = sess.run(model.u_batch_ratings, {model.users: [u], model.u_pos_items: test_items,
                                                        model.mess_dropout: [0.] * len(eval(args.layer_size)),
                                                        model.node_dropout: [0.]*len(eval(args.layer_size))})
            map_item_score = {}
            for i in range(len(test_items)):
                map_item_score[test_items[i]] = rate[0][i]
            ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
            hit = getHitRatio(ranklist, test_item)
            ndcg = getNDCG(ranklist, test_item)
            hits.append(hit)
            ndcgs.append(ndcg)
    result['ndcg'] = np.array(ndcgs).mean()
    result['hit_ratio'] = np.array(hits).mean()

    return result