from visualization.receptive_field import receptive_field
from models.deeperEmbedded import DeeperEmbededNet
from options.options_setting import Opt

opt = Opt()
opt.drop_prob = 0.3
opt.use_minmax_scale = False
opt.use_extra = True
opt.add_noise_prob = 0
opt.window_size = 3000
opt.extra_length = 5
opt.input_nc = 5
opt.input_length = 3000
model = DeeperEmbededNet(opt)
print(receptive_field(model, input_size=(5, 3000), batch_size=16, device='cpu'))
