from __future__ import print_function, absolute_import
import os
import sys 
from PIL import Image
import numpy as np
from matplotlib import cm 

import torch
from torch.utils.data import Dataset
import random
import transforms as T

def read_image(img_path):
	"""Keep reading image until succeed.
	This can avoid IOError incurred by heavy IO process."""
	got_img = False
	while not got_img:
		try:
			img = Image.open(img_path).convert('RGB')
			img = img.resize((100,100),Image.ANTIALIAS)
			got_img = True
		except IOError:
			print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
			pass
	return img

def read_depth(img_depth):
	got_img = False
	while not got_img:
		try:
			img = np.load(img_depth,allow_pickle=True)
			img = Image.fromarray(np.uint8(cm.jet(img)*255))
			img = img.convert('RGB')
			img = img.resize((100,100),Image.ANTIALIAS)
			got_img = True
		except IOError:
			print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_depth))
			pass
	return img



class VideoDataset(Dataset):
	"""Video Person ReID Dataset.
	Note batch data has shape (batch, seq_len, channel, height, width).
	"""
	sample_methods = ['evenly', 'random', 'all']

	def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
		self.dataset = dataset
		self.seq_len = seq_len
		self.sample = sample
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img_paths, img_depths_paths, pid, camid = self.dataset[index]
		num = len(img_paths)
		num_d = len(img_depths_paths)
		if self.sample == 'random':
			"""
			Randomly sample seq_len consecutive frames from num frames,
			if num is smaller than seq_len, then replicate items.
			This sampling strategy is used in training phase.
			"""
			frame_indices = list(range(num))
			frame_indices_d = list(range(num_d))
			rand_end = max(0, len(frame_indices) - self.seq_len - 1)
			rand_end_d = max(0, len(frame_indices_d) - self.seq_len - 1)
			begin_index = random.randint(0, rand_end)
			begin_index_d = random.randint(0, rand_end_d)
			end_index = min(begin_index + self.seq_len, len(frame_indices))
			end_index_d = min(begin_index_d + self.seq_len, len(frame_indices_d))

			indices = frame_indices[begin_index:end_index]
			indices_d = frame_indices_d[begin_index_d:end_index_d]

			#Per rgb
			for index in indices:
				if len(indices) >= self.seq_len:
					break
				indices.append(index)
			#Per depth
			
			for index in indices_d:
				if len(indices_d) >= self.seq_len:
					break
				indices_d.append(index)
			indices=np.array(indices)
			indices_d=np.array(indices_d)
			
			imgs = []
			imgs_depth = []
			#Per RGB prendo ogni percorso file e carico immagine
			for index in indices:
				index=int(index)
				img_path = img_paths[index]
				img = read_image(img_path)
				if self.transform is not None:
					img = self.transform(img)
				img = img.unsqueeze(0)
				imgs.append(img)
				
			for index in indices_d:
				index=int(index)
				img_path = img_depths_paths[index]
				img = read_depth(img_path) #uso funz definita prima
				if self.transform is not None:
					img = self.transform(img) #applico sequenza di operazioni con i tensori def in transform che passo a Videoloader
				img = img.unsqueeze(0)
				imgs_depth.append(img)
				
			imgs = torch.cat(imgs, dim=0)
			imgs_depth = torch.cat(imgs_depth, dim=0)
			#imgs=imgs.permute(1,0,2,3)
			return imgs, imgs_depth, pid, camid #per ora passo solo il path, poi si dovra scrivere codice per caricare effettivamente immagini

		elif self.sample == 'dense':
			"""
			Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
			This sampling strategy is used in test phase.
			"""
			cur_index=0
			cur_index_d=0
			frame_indices = list(range(num))
			frame_indices_d = list(range(num_d))
			indices_list=[]
			indices_list_d=[]
			#Per rgb
			while num-cur_index > self.seq_len:
				indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
				cur_index+=self.seq_len

			#Per depth
			
			while num_d-cur_index_d > self.seq_len:
				indices_list_d.append(frame_indices_d[cur_index_d:cur_index_d+self.seq_len])
				cur_index_d+=self.seq_len
				
			last_seq=frame_indices[cur_index:]
			last_seq_d=frame_indices_d[cur_index_d:]

			#RGB
			for index in last_seq:
				if len(last_seq) >= self.seq_len:
					break
				last_seq.append(index)
			#Depth 
			
			for index in last_seq_d:
				if len(last_seq_d) >= self.seq_len:
					break
				last_seq_d.append(index)
				
			indices_list.append(last_seq)
			indices_list_d.append(last_seq_d)
			imgs_list=[]
			imgs_list_d=[]
			#Per RGB 
			for indices in indices_list:
				imgs = []
				for index in indices:
					index=int(index)
					img_path = img_paths[index]
					img = read_image(img_path)
					if self.transform is not None:
						img = self.transform(img)
					img = img.unsqueeze(0)
					imgs.append(img)
				imgs = torch.cat(imgs, dim=0)
				#imgs=imgs.permute(1,0,2,3)
				imgs_list.append(imgs)
			#Per Depth
			
			for indices_d in indices_list_d:
				imgs_d = []
				for index in indices_d:
					index=int(index)
					img_path = img_depths_paths[index]
					img = read_depth(img_path)
					if self.transform is not None:
						img = self.transform(img)
					img = img.unsqueeze(0)
					imgs_d.append(img)
				imgs_d = torch.cat(imgs_d, dim=0)
				#imgs=imgs.permute(1,0,2,3)
				imgs_list_d.append(imgs_d)
			
			imgs_array = torch.stack(imgs_list)
			imgs_array_d = torch.stack(imgs_list_d)
			return imgs_array, imgs_array_d, pid, camid #idem come sopra 

		else:
			raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))







