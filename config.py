

class Config(object):
    def __init__(self):
        self.device = 'cuda'

        self.init_lr = 1.0e-3
        self.lr_steps = 100 #decrease at every 100th epoch
        self.b1 = 0.9
        self.b2 = 0.999

        self.n_epochs = 200

        self.resume = False
