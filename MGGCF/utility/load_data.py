import numpy as np
import random as rd
import scipy.sparse as sp
import re
from time import time

class Data(object):
    def __init__(self, path, batch_size, num_negatives):
        self.path = path
        self.batch_size = batch_size
        self.num_negatives = num_negatives

        group_train_file = path + '/groupRatingTrain.txt'
        group_test_file = path + '/groupRatingTest.txt'
        group_neg_file = path + '/groupRatingNegative.txt'
        user_train_file = path + '/userRatingTrain.txt'
        user_test_file = path + '/userRatingTest.txt'
        user_neg_file = path + '/userRatingNegative.txt'
        group_file = path + '/groupMember.txt'
        with open(group_file,"r",encoding='UTF-8') as f:
            group_num = len(f.readlines())

        self.n_groups = group_num


        # user data
        self.R_user_item,self.user_train_items,self.n_user_train = self.load_rating_file_as_matrix(user_train_file)
        self.user_test_set,self.n_user_test = self.load_rating_file_as_dict(user_test_file)
        self.user_neg_pools = self.load_negative_file(user_neg_file)
        self.n_users, self.n_items = self.R_user_item.shape
        # group data
        self.R_group_item,self.group_train_items,self.n_group_train = self.load_group_rating_file_as_matrix(group_train_file)
        self.group_test_set,self.n_group_test = self.load_rating_file_as_dict(group_test_file)
        self.group_neg_pools = self.load_negative_file(group_neg_file)
        self.print_statistics()

        #group_user matrix
        self.R_group_user = self.get_group_user_mat(group_file)
        self.user_instance_dict = self.get_train_instance(self.R_user_item)
        self.group_instance_dict =self.get_train_instance(self.R_group_item)




    def load_group_rating_file_as_matrix(self,filename):
        mat = sp.dok_matrix((self.n_groups, self.n_items), dtype=np.float32)
        n_train = 0
        train_items = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                        if user not in train_items.keys():
                            train_items[user] = [item]
                        else:
                            train_items[user].append(item)
                        n_train+=1

                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                    n_train+=1
                    if user not in train_items.keys():
                        train_items[user] = [item]
                    else:
                        train_items[user].append(item)
                line = f.readline()
        return mat,train_items,n_train

    def get_group_user_mat(self,filename):
        mat = sp.dok_matrix((self.n_groups, self.n_users), dtype=np.float32)
        with open(filename,"r",encoding='UTF-8') as f:
            line = f.readline().strip()
            while line != None and line != "":
                arr = line.split(" ")
                group = int(arr[0])
                users = [int(i) for i in arr[1].split(',')]
                for user in users:
                    mat[group,user] = 1.0
                line = f.readline().strip()
        return mat

    def load_rating_file_as_dict(self, filename):
        ratingdict = {}
        n_test = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                n_test += 1
                if user not in ratingdict.keys():
                    ratingdict[user] = [item]
                else:
                    ratingdict[user].append(item)
                line = f.readline()
        return ratingdict,n_test

    def load_negative_file(self, filename):
        neg_pools = {}
        with open(filename, "r") as f:
            line = f.readline().strip()
            while line != None and line != "":
                arr = line.split(" ")
                (user,pos_item) = eval(arr[0])
                neg_items = [int(i) for i in arr[1:]]
                neg_pools[(user,pos_item)] = neg_items
                line = f.readline().strip()
        return neg_pools

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        n_train = 0
        train_items = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                        if user not in train_items.keys():
                            train_items[user] = [item]
                        else:
                            train_items[user].append(item)
                        n_train+=1

                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                    n_train+=1
                    if user not in train_items.keys():
                        train_items[user] = [item]
                    else:
                        train_items[user].append(item)
                line = f.readline()
        return mat,train_items,n_train

    def readnum(self,filename):
        with open(filename,"r",encoding='iso8859-1') as f:
            lastline = f.readlines()[-1].split("::")
            num = lastline[0]
        return num

    def get_adj_mat(self,graph_type):
        try:
            t1 = time()
            norm_adj_mat = sp.load_npz(self.path + '/'+ str(graph_type) + '_s_norm_adj_mat.npz')
            print('already load adj matrix', norm_adj_mat.shape, time() - t1)
        except Exception:
            norm_adj_mat = self.create_adj_mat(graph_type)
            sp.save_npz(self.path + '/'+ str(graph_type) + '_s_norm_adj_mat.npz', norm_adj_mat)
        return norm_adj_mat

    def create_adj_mat(self,graph_type):
        t1 = time()
        if graph_type == 0:
            adj_mat = sp.dok_matrix((self.n_users + self.n_items,self.n_users + self.n_items),dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R_user_item.tolil()
            adj_mat[:self.n_users,self.n_users:] = R
            adj_mat[self.n_users:,:self.n_users] = R.T
        elif graph_type ==1:
            adj_mat = sp.dok_matrix((self.n_groups + self.n_items, self.n_groups + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R_group_item.tolil()
            adj_mat[:self.n_groups,self.n_groups:] = R
            adj_mat[self.n_groups:, :self.n_groups] = R.T
        else:
            adj_mat = sp.dok_matrix((self.n_groups + self.n_users, self.n_groups + self.n_users), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R_group_user.tolil()
            adj_mat[:self.n_groups, self.n_groups:] = R
            adj_mat[self.n_groups:, :self.n_groups] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        print('already normalize adjacency matrix', time() - t2)
        return norm_adj_mat.tocsr()

    def get_num_users_items_groups(self):
        return self.n_users, self.n_items, self.n_groups

    def print_statistics(self):
        print('n_users=%d, n_items=%d, n_groups = %d' % (self.n_users, self.n_items, self.n_groups))
        print('n_user_interactions=%d' % (self.n_user_train + self.n_user_test))
        print('n_group_interactions=%d' % (self.n_group_train +self.n_group_test))
        print('n_user_train=%d, n_user_test=%d, user_sparsity=%.5f' % (self.n_user_train, self.n_user_test, (self.n_user_train + self.n_user_test)/(self.n_users * self.n_items)))
        print('n_group_train=%d, n_group_test=%d, group_sparsity=%.5f' % (
        self.n_group_train, self.n_group_test, (self.n_group_train + self.n_group_test) / (self.n_groups * self.n_items)))

    def u_sample(self,batch_size):

        pos_item_input = []
        neg_item_input = []
        users = [i for i in range(self.n_users)]
        if batch_size< self.n_users:
            input_users = rd.sample(users,batch_size)
        else:
            input_users = [rd.choice(users) for _ in range(batch_size)]
        for user in input_users:
            pos_item_dict = rd.choice(self.user_instance_dict[user])
            pos_item = list(pos_item_dict.keys())[0]
            neg_items = pos_item_dict[pos_item]
            neg_item = rd.choice(neg_items)
            pos_item_input.append(pos_item)
            neg_item_input.append(neg_item)
        return input_users, pos_item_input, neg_item_input

    def g_sample(self,batch_size):
        pos_item_input = []
        neg_item_input = []
        groups = [i for i in range(self.n_groups)]
        if batch_size < self.n_groups:
            input_groups = rd.sample(groups, batch_size)
        else:
            input_groups = [rd.choice(groups) for _ in range(batch_size)]
        for group in input_groups:
            pos_item_dict = rd.choice(self.group_instance_dict[group])
            pos_item = list(pos_item_dict.keys())[0]
            neg_items = pos_item_dict[pos_item]
            neg_item = rd.choice(neg_items)
            pos_item_input.append(pos_item)
            neg_item_input.append(neg_item)
        return input_groups, pos_item_input, neg_item_input

    def get_train_instance(self,train_matrix):
        instance_dict = {}
        for (user, pos_item) in train_matrix.keys():
            if user not in instance_dict.keys():
                neg_items = []
                for _ in range(self.num_negatives):
                    neg_item = np.random.randint(self.n_items)
                    while (user, neg_item) in train_matrix.keys():
                        neg_item = np.random.randint(self.n_items)
                    neg_items.append(neg_item)
                temp_dict = {pos_item:neg_items}
                instance_dict[user]=[temp_dict]
            else:
                neg_items = []
                for _ in range(self.num_negatives):
                    neg_item = np.random.randint(self.n_items)
                    while (user, neg_item) in train_matrix.keys():
                        neg_item = np.random.randint(self.n_items)
                    neg_items.append(neg_item)
                temp_dict = {pos_item: neg_items}
                instance_dict[user].append(temp_dict)
        return instance_dict

















