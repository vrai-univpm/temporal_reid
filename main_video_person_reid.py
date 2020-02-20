from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import data_manager
from video_loader import VideoDataset
import transforms as T
import models
from models import resnet3d
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from datetime import datetime as dt
#from torchsummary import summary


parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='dataset', choices=data_manager.get_names())
parser.add_argument('--data_path', default="D:/Downloads/TVPR2-150-Orig", help="Path for the dataset")
parser.add_argument('-j', '--workers', default=4, type=int, help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=100, help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=100, help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=5, help="number of images to sample in a tracklet default 4")
parser.add_argument('--num-instances', type=int, default=4, help="number of instances per identity")
parser.add_argument('--id_train', default="1-21", help="IDs for training")
parser.add_argument('--id_test', default="101-121", help="IDs for test (query/gallery)")
parser.add_argument('--clip_unbalanced', action='store_true', default=False, help="clip_unbalanced")

# Optimization options
parser.add_argument('--max-epoch', default=100, type=int, help="maximum epochs to run default 800")
parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=8, type=int, help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int, help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")

parser.add_argument('--htri-only', action='store_true', default=False, help="if this is True, only htri loss is used in training default false")
parser.add_argument('--use_depth', action='store_true', default=False, help="if this is True, use depth channel for training")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50tp', help="resnet503d, resnet50tp, resnet50ta, resnet50rnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
parser.add_argument('--print-freq', type=int, default=80, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='C:/Users/megis/PycharmProjects/progetti-python3/reid/resnet-50-kinetics.pth', help='need to be set for resnet3d models')
#parser.add_argument('--pretrained-model', type=str, default='D:/Re-ID test/TP100d_ep1000.pth.tar', help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only", default=False)
parser.add_argument('--eval-step', type=int, default=5, help="run evaluation for every N epochs default 50  (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
	print("INIT: {}".format(dt.now()))
	torch.manual_seed(args.seed)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
	use_gpu = torch.cuda.is_available()
	if args.use_cpu: use_gpu = False
	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	if not args.evaluate:
		sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
	else:
		sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
	print("==========\nArgs:{}\n==========".format(args))

	if use_gpu:
		print("Currently using GPU {}".format(args.gpu_devices))
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)
	else:
		print("Currently using CPU (GPU is highly recommended)")

	print("Initializing dataset: {}".format(args.dataset))
	dataset = data_manager.init_dataset(name=args.dataset, data_path=args.data_path, id_train=args.id_train, id_test=args.id_test, unbalanced=args.clip_unbalanced, seq_len=args.seq_len)

	transform_train = T.Compose([
		T.Random2DTranslation(args.height, args.width),
		T.RandomHorizontalFlip(),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	transform_test = T.Compose([
		T.Resize((args.height, args.width)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	pin_memory = True if use_gpu else False


	trainloader = DataLoader(
		VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
		sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
		batch_size=args.train_batch, num_workers=args.workers,
		pin_memory=pin_memory, drop_last=True,
	)
	
	

	queryloader = DataLoader(
		VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
		batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
		pin_memory=pin_memory, drop_last=False,
	)


	galleryloader = DataLoader(
		VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
		batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
		pin_memory=pin_memory, drop_last=False,
	)
	

	print("Initializing model: {} - {}".format(args.arch,dt.now()))
	print("- use_depth={}".format(args.use_depth))
	if args.arch=='resnet503d':
		#model = resnet3d.resnet50(num_classes=dataset.num_train_pids, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len)
		model = resnet3d.resnet50(num_classes=dataset.num_train_pids, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len, loss={'xent', 'htri'})
		if not os.path.exists(args.pretrained_model):
			raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
		print("Loading checkpoint from '{}'".format(args.pretrained_model))
		checkpoint = torch.load(args.pretrained_model)
		state_dict = {}
		for key in checkpoint['state_dict']:
			if 'fc' in key: continue
			state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
		model.load_state_dict(state_dict, strict=False)
	else:
		model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
	criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
	criterion_htri = TripletLoss(margin=args.margin)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	if args.stepsize > 0:
		scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
	start_epoch = args.start_epoch

	if use_gpu:
		model = nn.DataParallel(model).cuda()

	if args.evaluate:
		print("Evaluate only")
		print("Loading checkpoint from '{}'".format(args.pretrained_model))
		#model.load_state_dict(torch.load(args.pretrained_model))
		checkpoint = torch.load(args.pretrained_model)
		state_dict = {}
		for key in checkpoint['state_dict']:
			#if 'fc' in key: continue
			state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
		model.load_state_dict(state_dict, strict=False)
		test(model, queryloader, galleryloader, args.pool, use_gpu)
		return

	start_time = time.time()
	best_rank1 = -np.inf
	if args.arch=='resnet503d':
		torch.backends.cudnn.benchmark = False

	plotx=[]
	ploty=[]
	for epoch in range(start_epoch, args.max_epoch):
		print("==> Epoch {}/{}  -  {}".format(epoch+1, args.max_epoch, dt.now()))
		
		train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
		
		if args.stepsize > 0: scheduler.step()
		
		if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
			print("==> Test  -  {}".format(dt.now()))
			rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
			ploty.append(rank1)
			plotx.append(epoch+1)
			is_best = rank1 > best_rank1
			if is_best: best_rank1 = rank1

			if use_gpu:
				state_dict = model.module.state_dict()
			else:
				state_dict = model.state_dict()
			save_checkpoint({
				'state_dict': state_dict,
				'rank1': rank1,
				'epoch': epoch,
			}, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

	elapsed = round(time.time() - start_time)
	elapsed = str(datetime.timedelta(seconds=elapsed))
	print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
	plt.plot(plotx, ploty)
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.savefig(args.save_dir+'/test.png')

def train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
	model.train()
	losses = AverageMeter()
	
	for batch_idx, (imgs, imgs_depths, pids, _) in enumerate(trainloader):
		if use_gpu:
			imgs, imgs_depths, pids = imgs.cuda(),imgs_depths.cuda(), pids.cuda()
		imgs,imgs_depths, pids = Variable(imgs), Variable(imgs_depths), Variable(pids)

		#print(summary(model, [imgs.shape, imgs_depths.shape]))
		#input("wait...")

		outputs, features, features_d = model(imgs, imgs_depths, args.use_depth)
		if args.htri_only:
			# only use hard triplet loss to train the network
			loss = criterion_htri(features, pids)
			#loss_d = criterion_htri(features_d, pids)
		else:
			# combine hard triplet loss with cross entropy loss
			xent_loss = criterion_xent(outputs, pids)
			htri_loss = criterion_htri(features, pids)
			#htri_loss_d = criterion_htri(features_d, pids)
			#htri_loss_media = (htri_loss + htri_loss_d)/2 
			loss = xent_loss + htri_loss 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#losses.update(loss.data[0], pids.size(0))	#Modificato per PyTOrch aggiornato
		losses.update(loss.item() , pids.size(0))
		#print("Training rgb ok")

		if (batch_idx+1) % args.print_freq == 0:
			print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))
	print("Total Batches={}".format(batch_idx+1))

def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
	model.eval()
	with torch.no_grad():
		qf, qf_d, q_pids, q_camids = [], [], [], []
		for batch_idx, (imgs, imgs_depths, pids, camids) in enumerate(queryloader):
			if use_gpu:
				imgs = imgs.cuda()
				imgs_depths = imgs_depths.cuda()
			imgs = Variable(imgs)	#, volatile=True)
			imgs_depths = Variable(imgs_depths)		#, volatile=True)
			# b=1, n=number of clips, s=16
			b, n, s, c, h, w = imgs.size()
			bd, nd, sd, cd, hd, wd = imgs_depths.size()
			assert(b==1)
			assert(bd==1)
			imgs = imgs.view(b*n, s, c, h, w)
			imgs_depths = imgs_depths.view(bd*nd, sd, cd, hd,wd)
			#features = model(imgs)
			features = model(imgs, imgs_depths, args.use_depth)
			features = features.view(n, -1)
			features = torch.mean(features, 0)
			features = features.data.cpu()
			#features_d = features_d.view(nd, -1)
			#features_d = torch.mean(features_d, 0)
			#features_d = features_d.data.cpu()
			qf.append(features)
			#qf_d.append(features_d)
			q_pids.extend(pids)
			q_camids.extend(camids)
		qf = torch.stack(qf)
		#qf_d =torch.stack(qf_d)
		q_pids = np.asarray(q_pids)
		q_camids = np.asarray(q_camids)

		print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

		gf, gf_d, g_pids, g_camids = [], [], [], []
		for batch_idx, (imgs, imgs_depths, pids, camids) in enumerate(galleryloader):
			if use_gpu:
				imgs = imgs.cuda()
				imgs_depths = imgs_depths.cuda()
			imgs = Variable(imgs)	#, volatile=True)
			imgs_depths = Variable(imgs_depths)		#, volatile=True)
			b, n, s, c, h, w = imgs.size()
			bd, nd, sd, cd, hd, wd = imgs_depths.size()
			imgs = imgs.view(b*n, s , c, h, w)
			imgs_depths = imgs_depths.view(bd*nd, sd, cd, hd, wd)
			assert(b==1)
			assert(bd==1)
			#features = model(imgs)
			features = model(imgs, imgs_depths, args.use_depth)
			features = features.view(n, -1)
			#features_d = features_d.view(nd, -1)

			if pool == 'avg':
				features = torch.mean(features, 0)
				#features_d = torch.mean(features_d, 0)
			else:
				features, _ = torch.max(features, 0)
				#features_d, _ = torch.max(features_d, 0)
			features = features.data.cpu()
			#features_d = features_d.data.cpu()
			gf.append(features)
			#gf_d.append(features_d)
			g_pids.extend(pids)
			g_camids.extend(camids)
		gf = torch.stack(gf)
		#gf_d =torch.stack(gf_d)
		g_pids = np.asarray(g_pids)
		g_camids = np.asarray(g_camids)

		print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
		print("Computing distance matrix")

		#m, n, md, nd = qf.size(0), gf.size(0),qf_d.size(0),gf_d.size(0) #rimettere depth
		m, n = qf.size(0), gf.size(0)  # rimettere depth
		distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
		distmat.addmm_(1, -2, qf, gf.t())
		distmat = distmat.numpy()
		#distmat_d = torch.pow(qf_d, 2).sum(dim=1, keepdim=True).expand(md, nd) + torch.pow(gf_d, 2).sum(dim=1, keepdim=True).expand(nd, md).t()
		#distmat_d.addmm_(1, -2, qf_d, gf_d.t())
		#distmat_d = distmat_d.numpy()
		#distmat_tot = (distmat + distmat_d)/2

		print("Computing CMC and mAP")
		#cmc, mAP = evaluate(distmat_tot, q_pids, g_pids, q_camids, g_camids) #rimettere distmat_tot
		cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)  # rimettere distmat_tot

		print("Results ----------")
		print("mAP: {:.1%}".format(mAP))
		print("CMC curve")
		for r in ranks:
			print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
		print("------------------")

		return cmc[0]

if __name__ == '__main__':
	main()