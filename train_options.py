"""Training options."""

import argparse

parser = argparse.ArgumentParser()

######## Data ########
parser.add_argument('--train', required=True, type=str,
                    help='Path to training data.')
parser.add_argument('--eval', required=True, type=str,
                    help='Path to eval data.')
# (distance1, distance2) should be (1, 2), (3, 3), or (6, 6).
# They correspond to the 3 levels of hierarchy described in paper.
parser.add_argument('--distance1', type=int,
                    help='Distance to left interpolation source.')
parser.add_argument('--distance2', type=int,
                    help='Distance to right interpolation source.')
parser.add_argument('--train-mv', type=str,
                    help='Path to motion vectors of training set.')
parser.add_argument('--eval-mv', type=str,
                    help='Path to motion vectors of evaluation set.')

######## Model ########
parser.add_argument('--v-compress', action='store_true',
                    help='True: video compression model. False: image compression.')
parser.add_argument('--iterations', type=int, default=10, 
                    help='# iterations of progressive encoding/decoding.')
parser.add_argument('--bits', default=16, type=int, 
                    help='Bottle neck size.')
parser.add_argument('--patch', default=64, type=int, 
                    help='Patch size.')
parser.add_argument('--shrink', type=int, default=2, 
                    help='Reducing # channels in U-net by this factor.')

# More model variants for ablation study. Please see paper for details.
parser.add_argument('--warp', action='store_true',
                    help='Whether to use motion information to warp U-net features.')
parser.add_argument('--fuse-encoder', action='store_true',
                    help='Whether to fuse context features into encoder.')
parser.add_argument('--encoder-fuse-level', type=int,
                    help='# encoder layers to fuse context information into.')
parser.add_argument('--decoder-fuse-level', type=int,
                    help='# decoder layers to fuse context information into.')
parser.add_argument('--stack', action='store_true', 
                    help='Whether to stack context frames as encoder input.')

######## Learning ########
parser.add_argument('--max-train-iters', type=int, default=100000,
                    help='Max training iterations.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='Learning rate.')
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient clipping.')
# parser.add_argument('--schedule', default='15000,40000,100000,250000', type=str,
#                     help='Schedule milestones.')
parser.add_argument('--schedule', default='50000,60000,70000,80000,90000', type=str,
                    help='Schedule milestones.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--batch-size', type=int, default=16, 
                    help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1,
                    help='Batch size for evaluation.')

# To save computation, we compute objective for multiple
# crops for each forward pass.
parser.add_argument('--num-crops', type=int, default=2,
                    help='# training crops per example.')
parser.add_argument('--gpus', default='0', type=str,
                    help='GPU indices separated by comma, e.g. \"0,1\".')

######## Experiment ########
parser.add_argument('--out-dir', type=str, default='output',
                    help='Output directory (for compressed codes & output images).')
parser.add_argument('--model-dir', type=str, default='model',
                    help='Path to model folder.')
parser.add_argument('--load-model-name', type=str,
                    help='Checkpoint name to load. (Do nothing if not specified.)')
parser.add_argument('--load-iter', type=int,
                    help='Iteraction of checkpoint to load.')
parser.add_argument('--save-model-name', type=str, default='demo',
                    help='Checkpoint name to save.')
parser.add_argument('--save-codes', action='store_true',
                    help='If true, write compressed codes during eval.')
parser.add_argument('--save-out-img', action='store_true',
                    help='If true, save output images during eval.')
parser.add_argument('--checkpoint-iters', type=int, default=10000,
                    help='Model checkpoint period.')
parser.add_argument('--eval-iters', type=int, default=4500,
                    help='Evaluation period.')
