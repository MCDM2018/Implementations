import argparse, os, torch
from GAN import GAN

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed','celeba'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=32, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0001)
    parser.add_argument('--lrD', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5) ### default 0.5
    parser.add_argument('--beta2', type=float, default=0.9) ### default 0.999
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--reg_lam',type=float,default=10)
    parser.add_argument('--gen_mode',type=bool,default=False)
    parser.add_argument('--eig_min',type=float,default=1)
    parser.add_argument('--eig_max',type=float,default=20)
    parser.add_argument('--eps',type=float,default=1)
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

        # declare instance for GAN
    if args.gan_type == 'GAN':
        gan = GAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

        # launch the graph in a session

    if args.gen_mode==False:
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch)
        print(" [*] Testing finished!")

    else:
        gan.load()
        gan.gen_mode(1000)

if __name__ == '__main__':
    main()
