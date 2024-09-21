import numpy as np



class CrossEntropy:
    def forward(self,ypred, ytrue):
        clip_ypred = np.clip(ypred,1e-7, 1- 1e-7) #prevent log(0)
        self.output = -np.sum(ytrue * np.log(clip_ypred))
