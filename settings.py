import os,time,socket,argparse

local = False

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')
    
def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_type", type=str, default='vgg16')
    argparser.add_argument("--criterion", type=str, default='ce')
    argparser.add_argument("--epochs", type=int, default=30)
    argparser.add_argument("--fake_num", type=int, default=3000)
    argparser.add_argument("--test_frac", type=float, default=.1)
    argparser.add_argument("--learning_rate", type=float, default=1e-3)
    argparser.add_argument("--weight_decay", type=float, default=1e-5)
    argparser.add_argument("--momentum", type=float, default=0.9)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--optim_type", type=str, default='adam',
                           help='sgd or adam')
    argparser.add_argument("--seed", type=int, default=123454321)
    argparser.add_argument("--k_fold", type=int, default=5)
    argparser.add_argument("--normalize", type=str2bool, default=True)
    argparser.add_argument("--batchnorm", type=str2bool, default=True)
    argparser.add_argument("--device_num", type=int, default='3',
                           help='cuda device number')
    argparser.add_argument("--output_path", type=str, default='/mnt/ssd1/sunyifan/WorkStation/PUF_Phenotype/out',
                           help='output path')
    argparser.add_argument("--path_prefix",type=str,default='/mnt/ssd1/sunyifan/WorkStation/PUF_Phenotype/Latency-DRAM-PUF-Dataset',help="PUF dataset path")
    argparser.add_argument("--model_path",type=str,default='/mnt/ssd1/sunyifan/WorkStation/PUF_Phenotype/out/',help="vgg16 saved model path")
   
    args = argparser.parse_args()

    return args

class Settings(object):
    '''
    Configuration for the project.
    '''
    def __init__(self):
        self.args = parse_arguments()
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        temp = os.path.join(self.args.output_path,f'seed_{self.args.seed}')
        if not os.path.exists(temp):
            os.makedirs(temp)
        temp = os.path.join(temp,"model_weights")
        if not os.path.exists(temp):
            os.makedirs(temp)
