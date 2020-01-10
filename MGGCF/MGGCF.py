import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from utility.helper import *
from utility.batch_test import *



class MGGCF(object):
    def __init__(self,data_config,pretrain_data):
        # argument settings
        self.model_type = 'MGGCF'

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_groups = data_config['n_groups']
        self.n_graphs = 3

        self.n_fold = 100

        self.norm_adj_gi = data_config['norm_adj_gi']
        self.norm_adj_ui = data_config['norm_adj_ui']
        self.norm_adj_gu = data_config['norm_adj_gu']

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)  #Output sizes of every layer[64,64,64]
        self.n_layers = len(self.weight_size)    #3

        self.model_type += '_l%d'%(self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]  #Regularizations hyperparameter
        self.verbose = args.verbose

        '''
            *********************************************************
            Create Placeholder for Input Data & Dropout.
        '''
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.u_pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.u_neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.groups = tf.placeholder(tf.int32,shape=(None,))
        self.g_pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.g_neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        """
            *********************************************************
            Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._build_weights()
        """
            *********************************************************
            Compute Graph-based Representations of all users & items & groups via Message-Passing Mechanism of Graph Neural Networks.
        """
        self.ui_embeddings, self.iu_embeddings = self._create_ngcf_embed(graph_type = 0)
        self.gi_embeddings, self.ig_embeddings = self._create_ngcf_embed(graph_type = 1)
        self.gu_embeddings, self.ug_embeddings = self._create_ngcf_embed(graph_type = 2)

        self.ua_embeddings = self.ui_embeddings + self.ug_embeddings
        self.ga_embeddings = self.gi_embeddings + self.gu_embeddings
        self.ia_embeddings = self.iu_embeddings + self.ig_embeddings

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """

        self.u_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_u_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.u_pos_items)
        self.neg_i_u_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.u_neg_items)
        self.g_embeddings = tf.nn.embedding_lookup(self.ga_embeddings, self.groups)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.g_pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.g_neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.g_batch_ratings = tf.matmul(self.g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                       transpose_b=True)
        self.u_batch_ratings = tf.matmul(self.u_embeddings, self.pos_i_u_embeddings, transpose_a=False,
                                         transpose_b=True)

        """
        *********************************************************
        Optimize via BPR loss
        """
        self.g_mf_loss, self.g_emb_loss, self.g_reg_loss = self.create_bpr_loss(self.g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.u_mf_loss, self.u_emb_loss, self.u_reg_loss = self.create_bpr_loss(self.u_embeddings,
                                                                                self.pos_i_u_embeddings,
                                                                                self.neg_i_u_embeddings)

        self.g_loss = self.g_mf_loss + self.g_emb_loss + self.g_reg_loss

        self.u_loss = self.u_mf_loss + self.u_emb_loss + self.u_reg_loss

        self.loss = self.g_loss + self.u_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _build_weights(self):
        print('Building Weights Phase')

        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        #initialize the embedding layer weights
        if self.pretrain_data is None:
            all_weights['group_embedding'] = tf.Variable(initializer([self.n_groups, self.emb_dim]))
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]))
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]))
            print('Using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            all_weights['group_embedding'] = tf.Variable(initial_value=self.pretrain_data['group_embed'],trainable=True,
                                                         name='group_embedding',dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        # initialize the embedding propagation layer weights

        for k in range(self.n_layers):
            all_weights['W_1_sum_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_1_gc_%d' % k)
            all_weights['b_1_sum_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_1_gc_%d' % k)
            all_weights['W_1_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_1_bi_%d' % k)
            all_weights['b_1_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_1_bi_%d' % k)

            all_weights['W_2_sum_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_1_gc_%d' % k)
            all_weights['b_2_sum_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_2_gc_%d' % k)
            all_weights['W_2_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_2_bi_%d' % k)
            all_weights['b_2_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_2_bi_%d' % k)

        return all_weights

    def _split_A_hat(self, X ,two_tuple):
        A_fold_hat = []
        fold_len = (two_tuple[0] + two_tuple[1]) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = two_tuple[0] + two_tuple[1]
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X, two_tuple):
        A_fold_hat = []

        fold_len = (two_tuple[0] + two_tuple[1]) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = two_tuple[0] + two_tuple[1]
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat
    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def _create_ngcf_embed(self,graph_type):

        if graph_type == 0:
            two_tuple = [self.n_users,self.n_items]
            ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
            X = self.norm_adj_ui
        elif graph_type ==1:
            two_tuple = [self.n_groups,self.n_items]
            ego_embeddings = tf.concat([self.weights['group_embedding'], self.weights['item_embedding']], axis=0)
            X = self.norm_adj_gi
        elif graph_type==2:
            two_tuple = [self.n_groups,self.n_users]
            ego_embeddings = tf.concat([self.weights['group_embedding'], self.weights['user_embedding']], axis=0)
            X = self.norm_adj_gu
        # Generate a set of adjacency sub-matrix for subgraph .
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(X,two_tuple)
        else:
            A_fold_hat = self._split_A_hat(X,two_tuple)

        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            if graph_type ==0 :
                sum_embeddings = tf.nn.leaky_relu(
                    tf.matmul(side_embeddings, self.weights['W_2_sum_%d' % k]) + self.weights['b_2_sum_%d' % k])#W_1
                # bi messages of neighbors.
                bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
                # transformed bi messages of neighbors.
                bi_embeddings = tf.nn.leaky_relu(
                    tf.matmul(bi_embeddings, self.weights['W_2_bi_%d' % k]) + self.weights['b_2_bi_%d' % k])
            else:
                sum_embeddings = tf.nn.leaky_relu(
                    tf.matmul(side_embeddings, self.weights['W_1_sum_%d' % k]) + self.weights['b_1_sum_%d' % k])
                # bi messages of neighbors.
                bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
                # transformed bi messages of neighbors.
                bi_embeddings = tf.nn.leaky_relu(
                    tf.matmul(bi_embeddings, self.weights['W_1_bi_%d' % k]) + self.weights['b_1_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        a_embeddings, b_embeddings = tf.split(all_embeddings, two_tuple, 0)
        return a_embeddings, b_embeddings

    def create_bpr_loss(self, groups, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(groups, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(groups, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(groups) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('Load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_groups'] = data_generator.n_groups
    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    norm_adj_ui = data_generator.get_adj_mat(graph_type = 0)
    norm_adj_gi = data_generator.get_adj_mat(graph_type = 1)
    norm_adj_gu = data_generator.get_adj_mat(graph_type = 2)



    config['norm_adj_gi'] = norm_adj_gi
    config['norm_adj_gu'] = norm_adj_gu
    config['norm_adj_ui'] = norm_adj_ui
    print('use the normalized adjacency matrix')


    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = MGGCF(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
        *********************************************************
        Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            groups_to_test = list(data_generator.group_test_set.keys())
            users_to_test = list(data_generator.user_test_set.keys())
            ret1 = test(sess, model, groups_to_test, drop_flag=True,test_type='group')
            ret2 = test(sess, model, users_to_test, drop_flag=True,test_type='user')

            cur_best_pre_0 = ret1['ndcg']

            pretrain_ret_g = 'pretrained model g_hit=[%.5f],g_ndcg=[%.5f]' % (ret1['hit_ratio'],ret1['ndcg'])
            pretrain_ret_u = 'pretrained model  u_hit=[%.5f],u_ndcg=[%.5f]' % (ret2['hit_ratio'],ret2['ndcg'])
            print(pretrain_ret_g)
            print(pretrain_ret_u)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('Without Pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')
    """
    *********************************************************
    Train.
    """



    loss_loger = []
    ndcg_loger_g, hit_loger_g = [], []
    ndcg_loger_u, hit_loger_u = [], []

    for epoch in range(args.epoch):
        t1 = time()
        loss, g_loss, u_loss = 0., 0., 0.

        n_batch = (data_generator.n_group_train + data_generator.n_user_train) // args.batch_size + 1
        for idx in range(n_batch):

            groups, g_pos_items, g_neg_items = data_generator.g_sample(args.batch_size)
            users, u_pos_items, u_neg_items = data_generator.u_sample(args.batch_size)
            _, batch_loss, batch_g_loss, batch_u_loss = sess.run([model.opt, model.loss, model.g_loss, model.u_loss],
                                                                     feed_dict={model.users: users, model.u_pos_items: u_pos_items, model.u_neg_items: u_neg_items,
                                                                                model.groups: groups, model.g_pos_items: g_pos_items, model.g_neg_items: g_neg_items,
                                                                                model.node_dropout: eval(args.node_dropout),
                                                                                model.mess_dropout:eval(args.mess_dropout)})
            loss += batch_loss
            g_loss += batch_g_loss
            u_loss += batch_u_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            continue


        if args.verbose > 0 and epoch % args.verbose == 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, g_loss, u_loss)
            print(perf_str)

        t2 = time()
        groups_to_test = list(data_generator.group_test_set.keys())
        users_to_test = list(data_generator.user_test_set.keys())
        ret1 = group_test(sess, model, groups_to_test, drop_flag=True)
        ret2 = user_test(sess, model, users_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        ndcg_loger_g.append(ret1['ndcg'])
        hit_loger_g.append(ret1['hit_ratio'])
        ndcg_loger_u.append(ret2['ndcg'])
        hit_loger_u.append(ret2['hit_ratio'])
        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], g_hit=[%.5f], g_ndcg=[%.5f], u_hit=[%.5f], u_ndcg=[%.5f]'% \
                    (epoch, t2 - t1, t3 - t2, loss, g_loss, u_loss, ret1['hit_ratio'],ret1['ndcg'], ret2['hit_ratio'],ret2['ndcg'])
            print(perf_str)

    ndcgs_g = np.array(ndcg_loger_g)
    hit_g = np.array(hit_loger_g)
    ndcgs_u = np.array(ndcg_loger_u)
    hit_u = np.array(hit_loger_u)
    save_path_1 = '%soutput/%s/%s.log_result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path_1)
    with open(save_path_1,'w') as f:
        np.savetxt(f, ndcgs_g, fmt='%f', delimiter=',')
        f.write("\n")
        np.savetxt(f, hit_g, fmt='%f', delimiter=',')
        f.write("\n")
        np.savetxt(f, ndcgs_u, fmt='%f', delimiter=',')
        f.write("\n")
        np.savetxt(f, hit_u, fmt='%f', delimiter=',')
        f.write("\n")

    best_rec_0 = max(ndcgs_g)
    idx = list(ndcgs_g).index(best_rec_0)
    final_perf = 'Best Iter=[%d]@[%.1f]\tg_hit=[%s], g_ndcg=[%s], u_hit=[%s], u_ndcg=[%s]' % \
                 (idx, time() - t0, '\t'.join(['%.5f' % hit_g[idx]]),
                  '\t'.join(['%.5f' % ndcgs_g[idx]]),
                  '\t'.join(['%.5f' % hit_u[idx]]),
                  '\t'.join(['%.5f' % ndcgs_u[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('k = %d, embed_size=%d, lr=%.4f, layer_size=%s, mess_dropout=%s, node_dropout = %s, regs=%s,adj_type=%s\n\t%s\n'
            % (args.K, args.embed_size, args.lr, args.layer_size, args.mess_dropout, args.node_dropout,args.regs,
           args.adj_type, final_perf))
    f.close()









