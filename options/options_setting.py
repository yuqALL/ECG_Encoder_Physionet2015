class Opt:
    is_init = False

    def __init__(self):
        self.base_setting()
        self.data_setting()
        self.signal_setting()
        self.is_init = True
        return

    def get_name(self):
        if not self.is_init:
            raise Exception("You should init configure.")
        name = str(3072 / 250) + 's_slice'
        if self.use_minmax_scale:
            name += '_norm'
        if self.add_noise_prob > 0:
            name += '_noise'
        if self.use_global_minmax:
            name += '_gnorm'
        if self.use_amplitude_noise:
            name += '_ampnoise'
        return name

    def base_setting(self):
        self.load_file_batch = 32
        self.batch_size = 256
        self.max_epoch = 200
        self.max_increase_epoch = 80
        self.np_to_seed = 1024  # random seed for numpy and pytorch
        self.drop_prob = 0.1
        self.debug = False
        self.training = True
        self.cuda = True
        self.lr = 1e-4
        self.weight_decay = 0

    def data_setting(self):
        self.data_folder = './data/training/'
        self.all_files = './data/training/RECORDS'

        self.exist_model_path = ''
        self.encoder_model_path = ''
        return

    def signal_setting(self):
        self.low_cut_hz = 0  # or 4
        self.SECOND_LENGTH = 300
        self.LONG_SECOND_LENGTH = 450
        self.load_sig_length = 15
        self.use_minmax_scale = False
        self.use_global_minmax = False
        self.use_gaussian_noise = False
        self.use_amplitude_noise = False
        self.add_noise_prob = 0.8

    def __repr__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
