from numpy.random import RandomState
import multiprocessing as mp
from datasets.data_read_utils import *

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def get_balanced_batches(
        n_trials, rng, shuffle, n_batches=None, batch_size=None
):
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches

    all_inds = np.array(range(n_trials))

    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    # for b in batches:
    #     yield b
    return batches


def stat_true_false(file_list, alarm_dict):
    true_list = []
    false_list = []
    for i, f in enumerate(file_list):
        if alarm_dict[f] == 0:
            true_list.append(i)
        elif alarm_dict[f] == 1:
            false_list.append(i)
        else:
            raise Exception("False Label Found!!!")
    return np.array(true_list), np.array(false_list)


class DatasetLoader:

    def __init__(self, opt):
        self.opt = opt
        self.rng = RandomState(20200426)
        self.files = load_file(self.opt.all_files, self.opt.data_folder)
        self.sig_data = dict()
        self.load_data(self.files)
        return

    def load_data(self, files):
        if not len(files):
            return
        for f in files:
            cnt = load_sig_data(f, True)
            self.sig_data[f] = cnt
        return

    def _load_batch_data(self, files, test_mode=False):
        if len(files) == 0:
            return None
        all_sig = []
        all_ori_sig = []
        for f in files:
            cnt = self.sig_data[f]
            if self.opt.add_noise_prob > 0 and not test_mode:
                origin_cnt, result_cnt = sub_window_split(cnt, test_mode=test_mode,
                                                          mmscale=self.opt.use_minmax_scale,
                                                          add_noise_prob=self.opt.add_noise_prob if not test_mode else 0)

                origin_cnt = np.concatenate((np.split(origin_cnt, origin_cnt.shape[2], axis=2)), axis=0).squeeze()[
                             :, :, np.newaxis]
                all_ori_sig.append(origin_cnt)
            else:
                result_cnt = sub_window_split(cnt, test_mode=test_mode,
                                              mmscale=self.opt.use_minmax_scale,
                                              add_noise_prob=self.opt.add_noise_prob if not test_mode else 0)
            n, _, k = result_cnt.shape
            result_cnt = np.concatenate((np.split(result_cnt, result_cnt.shape[2], axis=2)), axis=0).squeeze()[
                         :, :, np.newaxis]

            all_sig.append(result_cnt)
        if len(all_sig) == 0:
            print(files)
            return None, None

        sig = np.concatenate(np.array(all_sig, dtype=object), axis=0)
        sig = np.transpose(sig, (0, 2, 1))
        if self.opt.add_noise_prob > 0 and not test_mode:
            ori_sig = np.concatenate(np.array(all_ori_sig, dtype=object), axis=0)
            ori_sig = np.transpose(ori_sig, (0, 2, 1))
            return ori_sig, sig

        return None, sig

    def _load_batch_data_mp(self, files, test_mode=False):
        if len(files) == 0:
            return None
        all_sig = []
        n = len(files)
        sn = n // 4 + 1
        all_split_files = [files[i:i + sn] for i in range(0, n, sn)]
        import multiprocessing as mp
        pool = mp.Pool(4)
        results = [pool.apply_async(self._load_batch_data,
                                    args=(subfiles, test_mode)) for subfiles in all_split_files]
        # results = [p.get() for p in results]
        # results = np.concatenate(results, axis=0)
        return results

    def get_batches(self, dataset, shuffle, setname):
        if not isinstance(dataset, np.ndarray):
            dataset = np.array(dataset)
        if setname == "test":
            test_mode = True
        else:
            test_mode = False
        n_trials = len(dataset)
        people_batches = get_balanced_batches(
            n_trials, batch_size=self.opt.load_file_batch, rng=self.rng, shuffle=shuffle
        )
        for batch_inds in people_batches:
            batch = dataset[batch_inds]
            ori_sig, sig = self._load_batch_data(batch, test_mode=test_mode)
            data_batches = get_balanced_batches(
                sig.shape[0], batch_size=self.opt.batch_size, rng=self.rng, shuffle=True
            )
            for b_ind in data_batches:
                batch_X = sig[b_ind]
                if ori_sig is not None:
                    yield ori_sig[b_ind], batch_X
                else:
                    yield None, batch_X

    def mp_load(self, dataset, people_batches, test_mode=False):
        import multiprocessing as mp
        pool = mp.Pool(5)
        it = pool.imap_unordered(self._load_batch_data, [dataset[batch] for batch in people_batches])
        return it

    # def get_batches(self, dataset, shuffle, setname):
    #     if not isinstance(dataset, np.ndarray):
    #         dataset = np.array(dataset)
    #     if setname == "test":
    #         test_mode = True
    #     else:
    #         test_mode = False
    #     n_trials = len(dataset)
    #     people_batches = get_balanced_batches(
    #         n_trials, batch_size=self.opt.load_file_batch, rng=self.rng, shuffle=shuffle
    #     )
    #     sig_generator = self.mp_load(dataset, people_batches, test_mode)
    #     for sig in sig_generator:
    #         data_batches = get_balanced_batches(
    #             sig.shape[0], batch_size=self.opt.batch_size, rng=self.rng, shuffle=True
    #         )
    #         for b_ind in data_batches:
    #             batch_X = sig[b_ind]
    #             yield batch_X

    # def get_batches(self, dataset, shuffle, setname):
    #     if not isinstance(dataset, np.ndarray):
    #         dataset = np.array(dataset)
    #     if setname == "test":
    #         test_mode = True
    #     else:
    #         test_mode = False
    #     n_trials = len(dataset)
    #     people_batches = get_balanced_batches(
    #         n_trials, batch_size=self.opt.load_file_batch, rng=self.rng, shuffle=shuffle
    #     )
    #     import datetime
    #     for batch_inds in people_batches:
    #         batch = dataset[batch_inds]
    #         start_t = datetime.datetime.now()
    #         elapsed_sec = 0
    #         results = self._load_batch_data_mp(batch, test_mode=test_mode)
    #         end_t = datetime.datetime.now()
    #         elapsed_sec += (end_t - start_t).total_seconds()
    #         for p in results:
    #             start_t = datetime.datetime.now()
    #             sig = p.get()
    #             end_t = datetime.datetime.now()
    #             elapsed_sec += (end_t - start_t).total_seconds()
    #
    #             data_batches = get_balanced_batches(
    #                 sig.shape[0], batch_size=self.opt.batch_size, rng=self.rng, shuffle=True
    #             )
    #             for b_ind in data_batches:
    #                 batch_X = sig[b_ind]
    #                 yield batch_X
    #         print("加载一个batch需要消耗：" + "{:.2f}".format(elapsed_sec) + " 秒")


if __name__ == "__main__":
    from options.option_factory import gen_options
    import datetime

    opt = gen_options(True, False)
    opt.data_folder = '../data/training/'
    opt.all_files = '../data/training/test_files.txt'
    dataset = DatasetLoader(opt=opt)
    start_t = datetime.datetime.now()
    p1 = dataset._load_batch_data(dataset.files, False)
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("单线程计算 共消耗：" + "{:.2f}".format(elapsed_sec) + " 秒")

    start_t = datetime.datetime.now()
    p2 = dataset._load_batch_data_mp(dataset.files, False)
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多线程计算 共消耗：" + "{:.2f}".format(elapsed_sec) + " 秒")
