import time, os, torch, argparse, warnings, glob

#from dataLoader import train_loader, val_loader
from utils.tools import *
from talkNet import talkNet
from dataset import MyDataset
from torch.utils.data import DataLoader

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate') #0.0001
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=10,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=2500,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--windowSize',      type=float, default=25,  help='Number of frames of input winfow')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--savePath',     type=str, default="exps/exp1")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    args = parser.parse_args()
    # Data loader
    # if args.downloadAVA == True:
    #     preprocess_AVA(args)
    #     quit()

    # loader = train_loader(trialFileName = args.trainTrialAVA, \
    #                       audioPath      = os.path.join(args.audioPathAVA , 'train'), \
    #                       visualPath     = os.path.join(args.visualPathAVA, 'train'), \
    #                       **vars(args))
    # trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    # loader = val_loader(trialFileName = args.evalTrialAVA, \
    #                     audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
    #                     visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
    #                     **vars(args))
    # valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = 16)
    videoDir = "C:/Users/jmmol/Desktop/COSAS V7/TFM/npz"
    audioDir = "C:/Users/jmmol/Desktop/COSAS V7/TFM/mfccs"
    datasetTrain = MyDataset(int(args.windowSize),videoDir,audioDir,"trainSamples.csv")
    datasetTest = MyDataset(int(args.windowSize),videoDir,audioDir,"devSamples.csv")
    

    trainLoader = DataLoader(dataset=datasetTrain,shuffle=True,batch_size=32,num_workers=14) #Cambiar num_workers
    valLoader = DataLoader(dataset=datasetTest,shuffle=False,batch_size=32,num_workers=14) #Cambiar num_workers
    lr_ini = args.lr
    total_epoch = args.maxEpoch

    if args.evaluation == True:
        datasetTest = MyDataset(int(args.windowSize),videoDir,audioDir,"testSamples.csv")
        testLoader = DataLoader(dataset=datasetTest,shuffle=False,batch_size=32,num_workers=14) #Cambiar num_workers
        s = talkNet(**vars(args))
        s.load_parameters(r"C:\Users\jmmol\Desktop\COSAS V7\TFM\exps\exp1\model\model13_0006.model")
        #s.loadParameters(r'C:\Users\jmmol\Desktop\COSAS V7\TFM\exps\exp1\model\model21_0005.model')
        print("Model %s loaded from previous state!"%('pretrain_AVA.model'))
        testLoss, testACC, testmap = s.evaluate_network(loader = testLoader, **vars(args))
        print("Loss en test: %2.2f%%, ACC %2.2f%%, mAP: %2.2f%%"%(testLoss, testACC, testmap))
        quit()

    modelfiles = glob.glob('%s/model_0*.model'%(args.savePath+"/model"))
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = talkNet(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = talkNet(epoch = epoch, **vars(args))

    mAPs = []
    scoreFile = open(args.savePath+"/score.txt", "a+")

    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            s.save_parameters(args.savePath+"/model/model%d_%04d.model"%(int(args.windowSize),epoch))
            testLoss, testACC, testmap = s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args))
            mAPs.append(testmap)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("epoch %d, %d total epochs, LR %f, TRAINLOSS %f, TESTLOSS %f, testACC %2.2f%%, testmAP %2.2f%%, bestTestmAP %2.2f%%\n"%(epoch,total_epoch, lr_ini, loss,testLoss, testACC, mAPs[-1], max(mAPs)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            scoreFile.write("\n")
            scoreFile.flush()
            quit()

        epoch += 1

if __name__ == '__main__':
    main()
