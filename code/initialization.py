import os
import torch
import numpy as np
import torchaudio
import logging
import sys
from datetime import datetime


def initialization(args):
    # create the directory if not exist
    if not os.path.exists(args.logDir):
        os.mkdir(args.logDir)
        
    subDir = os.path.join(args.logDir, args.sub)
    if not os.path.exists(subDir):
        os.mkdir(subDir)        
        
    subDir2 = os.path.join(args.logDir, args.sub, args.logName)
    if not os.path.exists(subDir2):
        os.mkdir(subDir2)
        
    saveDir = os.path.join(args.logDir, args.sub, args.logName, args.task)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    args.savevoice = saveDir + '/eval_epovoice'
    if not os.path.exists(args.savevoice):
        os.mkdir(args.savevoice)
    
    if not os.path.exists(args.savevoice + '/train'):
        os.mkdir(args.savevoice + '/train')
    if not os.path.exists(args.savevoice + '/val'):
        os.mkdir(args.savevoice + '/val')
    if not os.path.exists(args.savevoice + '/test'):
        os.mkdir(args.savevoice + '/test')

    args.savefig = saveDir + '/eval_epofig'
    if not os.path.exists(args.savefig):
        os.mkdir(args.savefig)
        
    if not os.path.exists(args.savefig + '/train'):
        os.mkdir(args.savefig + '/train')
    if not os.path.exists(args.savefig + '/val'):
        os.mkdir(args.savefig + '/val')
    if not os.path.exists(args.savefig + '/test'):
        os.mkdir(args.savefig + '/test')
        
        
    args.savemodel = saveDir + '/savemodel'
    if not os.path.exists(args.savemodel):
        os.mkdir(args.savemodel)
        
    args.logs = saveDir + '/logs'
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
        
    # initialize logging handler
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M")
    log_file = os.path.join(args.logs, 'train_{}.log'.format(current_time))
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, 'w+'),
            logging.StreamHandler(sys.stdout)
        ])
    
    # Keep logging clean of pyxdf information
    logging.getLogger('pyxdf.pyxdf').setLevel(logging.WARNING)
    args.logger = logging.getLogger()
    
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    args.logger.info('Current cuda device: {} '.format(torch.cuda.current_device())) # check
    args.logger.info('The number of available GPU:{}'.format(torch.cuda.device_count()))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    args.logger.info('Config Information')
    
    
    # Dataset Define
    if args.dataset == "EEG_word":      # word: 64, sentence: 127, sEEG: 128
        args.in_ch = 64
        args.phon = False
        args.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    elif args.dataset == "EEG_sen":
        args.in_ch = 127
        args.phon = True
        args.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE 
    elif args.dataset == "sEEG":
        args.in_ch = 128
        args.phon = False
        args.bundle = torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_DE 
    else:
        raise NameError('Check Dataset Name')
        
    
    
    