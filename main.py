"""
main file

License is from https://github.com/aboulch/tec_prediction
"""

import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.ndimage import gaussian_filter
from datetime import date, timedelta


#####################
## COLOR PRINT
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

def printBlue(*args):
    print(bcolors.OKBLUE,*args,bcolors.ENDC)
def printGreen(*args):
    print(bcolors.OKGREEN,*args,bcolors.ENDC)
def printRed(*args):
    print(bcolors.FAIL,*args,bcolors.ENDC)
#######################



# data normalization
TEC_MEAN = 19
TEC_MIN = 0
TEC_MAX = 150

def preprocess(x):
    """Normalize TEC data."""
    return (x - TEC_MEAN) / (TEC_MAX - TEC_MIN)

def unprocess(x):
    """Unnormalize TEC data."""
    return x * (TEC_MAX - TEC_MIN) + TEC_MEAN

def get_input_targets(batch_np):
    """Separate input and target from sequence."""
    inputs_np =  batch_np[:args.seqStart]
    targets_np = batch_np[args.seqStart:]
    return inputs_np, targets_np

def get_periodic(inputs, prediction_len):
    """Get the part of the input corresponding to the prediction length."""
    # return inputs[-prediction_len:]
    if prediction_len>24:
        raise Exception("Error prediction > 48h, TODO")
    if prediction_len == 24:
        return inputs[-prediction_len:]
    else:
        return inputs[-24:-24+prediction_len]

def get_periodic_blur_targets_diff(periodic, targets):
    """Apply blur on sequence and compute difference (for residual learning)."""
    periodic_blur = periodic.copy()
    for i in range(periodic_blur.shape[0]):
        for j in range(periodic_blur.shape[1]):
            periodic_blur[i,j,0] = gaussian_filter(periodic_blur[i,j,0], sigma=3)
    targets_diff = targets - periodic_blur
    return periodic_blur, targets_diff

def rms(data, axis=None):
    """Compute RMS."""
    if(axis is None):
        return np.sqrt((data**2).mean())
    else:
        return np.sqrt((data**2).mean(axis=axis))



weights = np.arange(-36, 36).reshape((72, 1))
weights = np.repeat(weights, 72, 1)
weights = np.abs(weights)
weights[:36] -= 1
weights = np.cos((weights.astype(float) / 36) * np.pi / 2)
weights /= weights.sum()


def process_data(training=False):

    if training:
        # training mode
        net.train()
        #iterate on the train dataset
        t = tqdm(train_loader, ncols=150)
    else:
        net.eval()
        #iterate on the train dataset
        t = tqdm(test_loader, ncols=150)

    # define prediction length
    prediction_len = args.seqLength - args.seqStart

    loss = 0
    rms_ = 0 # mean rms oversequence
    rms_periodic = 0 # mean rms over sequence
    rms_per_frame = [0 for i in range(prediction_len)]
    rms_periodic_per_frame = [0 for i in range(prediction_len)]
    rms_per_sequence = []
    rms_per_sequence_periodic = []
    count = 0
    rms_lattitude = np.zeros(72)

    rms_global_mean = []

    for batch in t:
        count += batch[0].size(0) * prediction_len # count number of prediction images

        # preprocess the batch (TODO: go pytorch)
        batch_np = preprocess(batch[0].numpy().transpose((1,0,2,3,4)))

        # create inputs and targets for network
        inputs_np, targets_np = get_input_targets(batch_np)
        periodic_np = get_periodic(inputs_np, prediction_len)
        if args.diff:# use residual
            periodic_blur_np, targets_network_np = get_periodic_blur_targets_diff(periodic_np, targets_np)
            periodic_blur = torch.from_numpy(periodic_blur_np).float()
            if args.cuda:
                periodic_blur = periodic_blur.cuda()
        else:
            targets_network_np = targets_np.copy()
            periodic_blur = None

            periodic_blur_np, _ = get_periodic_blur_targets_diff(periodic_np, targets_np)
            periodic_blur = torch.from_numpy(periodic_blur_np).float()
            if args.cuda:
                periodic_blur = periodic_blur.cuda()

        # create pytorch tensors
        inputs = torch.from_numpy(inputs_np).float()
        targets = torch.from_numpy(targets_network_np).float()
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()


        if training:
            # set gradients to zero
            optimizer.zero_grad()
            # forward pass in the network
            outputs = net.forward(inputs, prediction_len, diff=args.diff, predict_diff_data=periodic_blur)
            # compute error and backprocj
            error = criterion(outputs, targets)
            error.backward()
            optimizer.step()
        else:
            outputs = net.forward(inputs, prediction_len, diff=args.diff, predict_diff_data=periodic_blur)
            error = criterion(outputs, targets) # compute loss for comparison


        # outputs
        outputs_np = outputs.cpu().data.numpy()
        if args.diff:
            outputs_complete = unprocess(outputs_np + periodic_blur_np)
        else:
            outputs_complete = unprocess(outputs_np)
        periodic_complete = unprocess(periodic_np)
        targets_complete = unprocess(targets_np)

        # update loss
        loss += float(error.cpu().item())

        # compute the rms for each image
        rms_tec_images = rms(outputs_complete-targets_complete, axis=(2,3,4))
        rms_tec_images_periodic = rms(periodic_complete-targets_complete, axis=(2,3,4))

        rms_tec_images_lattitude = rms(outputs_complete-targets_complete, axis=(2,4))
        rms_lattitude += rms_tec_images_lattitude.sum(axis=(0,1))

        #rms_gm = outputs_complete.mean(axis=(2,3,4))-targets_complete.mean(axis=(2,3,4))
        rms_gm = (outputs_complete*weights[None,None,None,:,:]).sum(axis=(2,3,4)) - (targets_complete*weights[None,None,None,:,:]).sum(axis=(2,3,4))
        rms_gm = rms_gm.transpose(1,0)
        for i in range(rms_gm.shape[0]):
            rms_global_mean.append(rms_gm[i])

        # update global rms
        rms_ += rms_tec_images.sum()
        rms_periodic += rms_tec_images_periodic.sum()

        # update rms per seq frame
        for frame_id in range(prediction_len):
            rms_per_frame[frame_id] += rms_tec_images[frame_id].sum()
            rms_periodic_per_frame[frame_id] += rms_tec_images_periodic[frame_id].sum()

        for seq_id in range(rms_tec_images.shape[1]):
            rms_per_sequence.append(rms_tec_images[:,seq_id].mean())
            rms_per_sequence_periodic.append(rms_tec_images_periodic[:,seq_id].mean())


        # update TQDM
        t.set_postfix(Loss=loss/count, RMS=rms_/count, RMS_P=rms_periodic/count)

    rms_global_mean = np.array(rms_global_mean)

    print("RMS GLOBAL MEAN", rms_global_mean.shape, rms(rms_global_mean, axis=1).mean())

    loss = loss/count
    rms_ = rms_/count
    rms_lattitude = rms_lattitude / count
    rms_periodic = rms_periodic/count
    for frame_id in range(prediction_len):
        rms_per_frame[frame_id] /= count / prediction_len
        rms_periodic_per_frame[frame_id] /= count / prediction_len

    return loss, rms_, rms_periodic, rms_per_frame, rms_periodic_per_frame, rms_per_sequence, rms_per_sequence_periodic, rms_lattitude



parser = argparse.ArgumentParser()
parser.add_argument("--seqLength", type=int, default=60, help="train network or not")
parser.add_argument("--seqStart", type=int, default=36)
parser.add_argument("--batchSize", type=int, default=16, help="train network or not")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="simple")
parser.add_argument("--diff", action="store_true")
parser.add_argument("--target", type=str, default="./results", help="target directory")
parser.add_argument("--source", type=str, default="./source", help="source directory")
args = parser.parse_args()
epoch_max = args.epochs

print("Seq Length", args.seqLength)

# create the result directory
if not os.path.exists(args.target):
    os.makedirs(args.target)

# define optimization parameters
root_dir=args.source

# CUDA
if args.cuda:
    torch.backends.cudnn.benchmark = True


printBlue("Creating data loader...")
from data_loader import SequenceLoader
ds = SequenceLoader(root_dir, args.seqLength, training=True)
ds_val = SequenceLoader(root_dir, args.seqLength, training=False)
train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchSize, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batchSize, shuffle=False, num_workers=2)

printBlue("Creating network...")
if args.model=="simple":
    from network_simple import SimpleConvRecurrent
    net = SimpleConvRecurrent(1)
elif args.model=="unet":
    from network_unet import UnetConvRecurrent
    net = UnetConvRecurrent(1)
elif args.model=="dilation121":
    from network_dilation_121 import UnetConvRecurrent
    net = UnetConvRecurrent(1)
else:
    printRed("Error bad network")
    exit()
if args.cuda:
    net.cuda()

print("PARAMTERS")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(net))
# exit()


printBlue("Setting up the optimizer...")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

printBlue("Setting up the criterion...")
criterion = torch.nn.L1Loss()

if not args.test:
    printBlue("TRAINING")

    loss_table = []
    rms_table = []
    rms_periodic_table = []

    # iterate on epochs
    for epoch in range(epoch_max):

        printGreen("Epoch", epoch)

        # train
        loss, rms_, rms_periodic, rms_per_frame, rms_per_frame_periodic, _, _ = process_data(training=True)

        # save the model
        torch.save(net.state_dict(), os.path.join(args.target, "state_dict.pth"))

# Test mode
printBlue("TESTING")

printBlue("Loading model")
net.load_from_filename(os.path.join(args.target, "state_dict.pth"))

with torch.no_grad():
    loss, rms_, rms_periodic, rms_per_frame, rms_per_frame_periodic, rms_per_sequence, rms_per_sequence_periodic, rms_lattitude = process_data()

print("Testing loss", loss)
print("Mean RMS per frame", rms_)
print("Mean RMS per frame (periodic)", rms_periodic)


print("Writing logs")
logs = open(os.path.join(args.target, "test_logs_{}.txt".format(args.seqLength)), "w")
logs.write(str(loss)+" ")
logs.write(str(rms_)+" ")
logs.write(str(rms_periodic)+" ")
logs.close()

logs = open(os.path.join(args.target, "test_logs_per_frame_{}.txt".format(args.seqLength)), "w")
for frame_id in range(len(rms_per_frame)):
    logs.write(str(frame_id)+" ")
    logs.write(str(rms_per_frame[frame_id])+" ")
    logs.write(str(rms_per_frame_periodic[frame_id])+" \n")
logs.close()

logs = open(os.path.join(args.target, "test_logs_per_sequence_{}.txt".format(args.seqLength)), 'w')
for seq_id in range(len(rms_per_sequence)):
    logs.write(str(seq_id)+" ")
    logs.write(str(rms_per_sequence[seq_id])+" ")
    logs.write(str(rms_per_sequence_periodic[seq_id])+" \n")
logs.close()

logs = open(os.path.join(args.target, "test_logs_lattitude_{}.txt".format(args.seqLength)), 'w')
for l_id in range(rms_lattitude.shape[0]):
    logs.write(str(rms_lattitude[l_id])+" ")
logs.close()
