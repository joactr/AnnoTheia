import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(talkNet, self).__init__()        
        self.model = talkNetModel().cuda()
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def forward(self,x):
        with torch.no_grad():
            audioFeature, visualFeature = x
            #print(audioFeature.shape,visualFeature.shape)

            audioEmbed = self.model.forward_audio_frontend(audioFeature.cuda()) # feedForward
            visualEmbed = self.model.forward_visual_frontend(visualFeature.cuda())
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            scores,labels = self.lossAV.forward(outsAV)  # returns scores,labels  



        return scores,labels

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            #print(audioFeature.shape,visualFeature.shape)
            self.zero_grad()
            audioEmbed = self.model.forward_audio_frontend(audioFeature.cuda()) # feedForward
            visualEmbed = self.model.forward_visual_frontend(visualFeature.cuda())
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels.reshape((-1)).cuda() # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr

    def evaluate_network(self, loader, **kwargs):
        self.eval()
        windowSize = kwargs.get('windowSize',24)
        predScores, predLabels = [], []
        index, top1, loss = 0, 0, 0
        for num, (audioFeature, visualFeature, labels) in enumerate(tqdm.tqdm(loader)):
            with torch.no_grad():                
                audioEmbed = self.model.forward_audio_frontend(audioFeature.cuda()) # feedForward
                visualEmbed = self.model.forward_visual_frontend(visualFeature.cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                outsA = self.model.forward_audio_backend(audioEmbed)
                outsV = self.model.forward_visual_backend(visualEmbed)
                labels = labels.reshape((-1)).cuda() # Loss         
                nlossAV, predScore, predLabel, prec = self.lossAV.forward(outsAV, labels)    
                nlossA = self.lossA.forward(outsA, labels)
                nlossV = self.lossV.forward(outsV, labels)
                nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
                loss += nloss.detach().cpu().numpy()
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                predLabels.extend(predLabel.detach().cpu().numpy().astype(int))
                top1 += prec
                index += len(labels)

        precision_eval = 100 * (top1/index)
        print("TESTACC:",precision_eval)

        df = pd.read_csv("testSamples.csv")
        df = df.loc[df.index.repeat(windowSize)].reset_index(drop=True)
        df["pred"] = predLabels
        df["posScore"] = predScores
        df.index.name = 'uid'
        df.to_csv("testPreds.csv")

        cmd = "python -O get_map.py -p testPreds.csv"
        mAP = str(subprocess.check_output(cmd)).split(' ')[2][:5]
        if mAP[-1] == "%": mAP = mAP[:-1]
        mAP = float(mAP)
        return loss/num, precision_eval, mAP


    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
