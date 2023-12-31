# 根据自己算法需要设置ClientConfig中的参数，不必要的参数就不用写入了
class ClientConfig:
    def __init__(self, idx):
        self.idx = idx
        # self.params = None
        self.params_dict = None
        self.epoch_idx = 0
        self.local_updates = 0
        self.train_batch_size = 0
        self.download_compress_ratio = 0.5
        # self.params = None

        self.train_data_idxes = None
        # self.model_type = None
        # self.dataset_type = None
        # self.batch_size = None
        self.lr = None
        # self.train_loader = None
        # self.decay_rate = None
        # self.min_lr = None
        # self.epoch = None
        # self.momentum = None
        # self.weight_decay = None
        # self.local_steps = 20

        self.aggregate_weight = 0.1

        self.train_time = 0
        self.send_time = 0
        self.q_level    =16
