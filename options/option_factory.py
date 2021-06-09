from options.options_setting import Opt


def gen_options(use_norm=False, use_noise=False):
    opt = Opt()
    opt.drop_prob = 0.3
    opt.load_sig_length = 300

    if use_norm:
        opt.use_minmax_scale = True
    else:
        opt.use_minmax_scale = False

    if use_noise:
        opt.add_noise_prob = 0.8
    else:
        opt.add_noise_prob = 0

    return opt
