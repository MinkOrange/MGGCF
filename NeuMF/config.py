

class Config(object):
    def __init__(self):
        self.path = './data/ML-1M/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = "./data/ML-1M/groupMember.txt"
        self.factor_num = 64
        self.num_layers = 3
        self.epoch = 30
        self.num_negatives = 4
        self.batch_size = 2048
        self.lr = [0.000005, 0.000001, 0.0000005]
        self.drop_ratio = 0.2
        self.topK = 5
