# ref:https://github.com/ShunyuYao/DFA-NeRF
import argparse
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # self.parser.add_argument('--model_save_path', type=str, default='snapshot/small_filter_wo_ct_wi_bn/real_data/combine/', help='path')
        self.parser.add_argument('--model_save_path', type=str, default='snapshot/version1/', help='path')
        self.parser.add_argument('--num_threads', type=int, default=2, help='number of threads')
        self.parser.add_argument('--max_dataset_size', type=int, default=150000, help='max dataset size')

        self.parser.add_argument('--n_epochs', type=int, default=40000, help='number of iterations')
        self.parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
        self.parser.add_argument('--init_type', type=str, default='uniform', help='[uniform | xavier]')
        self.parser.add_argument('--frequency_print_batch', type=int, default=1000, help='print messages every set iter')
        self.parser.add_argument('--frequency_save_model', type=int, default=2000, help='save model every set iter')
        self.parser.add_argument('--small', type=bool, default=True, help='use small model')
        self.parser.add_argument('--use_batch_norm', action='store_true', help='')
        self.parser.add_argument('--smooth_2nd', type=bool, default=True, help='')
        

        #loss weight for Gauss-Newton optimization
        self.parser.add_argument('--lambda_2d', type=float, default=0.001, help='weight of 2D projection loss')
        self.parser.add_argument('--lambda_depth', type=float, default=1.0, help='weight of depth loss')
        self.parser.add_argument('--lambda_reg', type=float, default=1.0, help='weight of regularization loss')
        
        self.parser.add_argument('--num_adja', type=int, default=6, help='number of nodes who affect a point')
        self.parser.add_argument('--num_corres', type=int, default=20000, help='number of corres')
        self.parser.add_argument('--iter_num', type=int, default=3, help='GN iter num')
        self.parser.add_argument('--width', type=int, default=512, help='image width')#480
        self.parser.add_argument('--height', type=int, default=512, help='image height')#640
        self.parser.add_argument('--crop_width', type=int, default=240, help='image width')
        self.parser.add_argument('--crop_height', type=int, default=320, help='image height')
        self.parser.add_argument('--max_num_edges', type=int, default=30000, help='number of edges')
        self.parser.add_argument('--max_num_nodes', type=int, default=1500, help='number of edges')
        self.parser.add_argument('--fdim', type=int, default=128)

        #loss weight for training
        self.parser.add_argument('--lambda_weights', type=float, default=0.0, help='weight of weights loss')#75
        self.parser.add_argument('--lambda_corres', type=float, default=1.0, help='weight of corres loss')#0, 1
        self.parser.add_argument('--lambda_graph', type=float, default=10.0, help='weight of graph loss')#1000, 5
        self.parser.add_argument('--lambda_warp', type=float, default=10.0, help='weight of warp loss')#1000, 5

        
    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        self.opt.isTest = self.isTest
        args = vars(self.opt)

        return self.opt

class TrainOptions(BaseOptions):
    # Override
    def initialize(self):
        BaseOptions.initialize(self)
        #syn_datasets/syn_new_train_data.txt
        self.parser.add_argument('--datapath', type=str, default='./data/train_data.txt', help='path')
        self.parser.add_argument('--pretrain_model_path', type=str, default='./pretrain_model/raft-small.pth', help='path')#
        self.parser.add_argument('--lr_C', type=float, default=0.00001, help='initial learning rate')#0.01
        self.parser.add_argument('--optimizer_C', type=str, default='sgd', help='[sgd | adam]')
        self.parser.add_argument('--lr_W', type=float, default=0.00001, help='initial learning rate')
        self.parser.add_argument('--lr_BSW', type=float, default=0.00001, help='initial learning rate')
        self.parser.add_argument('--optimizer_W', type=str, default='sgd', help='[sgd | adam]')
        self.parser.add_argument('--optimizer_BSW', type=str, default='sgd', help='[sgd | adam]')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=8000, help='multiply by a gamma every set iter')
        self.parser.add_argument('--lr_decay', type=float, default=0.1, help='coefficient of lr decay')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='0.0005coefficient of weight decay')
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        self.parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle data')

        self.parser.add_argument('--validation', type=str, nargs='+')
        #self.parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
        self.parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
        self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        self.parser.add_argument('--iters', type=int, default=12)

        self.parser.add_argument('--clip', type=float, default=1.0)
        self.parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
        self.parser.add_argument('--add_noise', action='store_true')

        self.parser.add_argument('--train_bsw', type=bool, default=True, help='whether to train bsw network')
        self.parser.add_argument('--train_weight', type=bool, default=True, help='whether to train weight network')
        self.parser.add_argument('--train_corres', type=bool, default=True, help='whether to train corresPred network')

        self.isTrain = True
        self.isTest = False

class ValOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        self.parser.add_argument('--datapath', type=str, default='./data/val_data.txt', help='path')
        self.parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle data')
        self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        self.parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        self.parser.add_argument('--iters', type=int, default=12)
        self.isTrain = True
        self.isTest = False

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--pretrain_model_path', type=str, default='./pretrain_model/raft-small.pth', help='path')#

        # self.parser.add_argument('--datapath', type=str, default='./data/real_train_data_1128_1.txt', help='path')
        # self.parser.add_argument('--datapath', type=str, default='./data_test_flow/test_data.txt', help='path')
        self.parser.add_argument('--savepath', type=str, default='flow_result',
                        help='save path')
        self.parser.add_argument('--datapath', type=str, default='/data_b/yudong/paper_code/TalkingHead-NeRF/data_guancha/guancha_flow.txt', 
            help='path')
        self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        self.parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        self.parser.add_argument('--iters', type=int, default=12)
        self.parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle data')
        self.isTrain = False
        self.isTest = True
