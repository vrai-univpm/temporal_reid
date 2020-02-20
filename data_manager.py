from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import random
import os.path as osp
from scipy.io import loadmat
import numpy as np

from utils import mkdir_if_missing, write_json, read_json

"""Dataset classes"""


class Mars(object):
	"""
	MARS
	Reference:
	Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
	
	Dataset statistics:
	# identities: 1261
	# tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
	# cameras: 6
	Args:
		min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
	"""
	root = './data/mars'
	train_name_path = osp.join(root, 'info/train_name.txt')
	test_name_path = osp.join(root, 'info/test_name.txt')
	track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
	track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
	query_IDX_path = osp.join(root, 'info/query_IDX.mat')

	def __init__(self, min_seq_len=0):
		self._check_before_run()

		# prepare meta data
		train_names = self._get_names(self.train_name_path)
		test_names = self._get_names(self.test_name_path)
		track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
		track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
		query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
		query_IDX -= 1 # index from 0
		track_query = track_test[query_IDX,:]
		gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
		track_gallery = track_test[gallery_IDX,:]

		train, num_train_tracklets, num_train_pids, num_train_imgs = \
		  self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

		query, num_query_tracklets, num_query_pids, num_query_imgs = \
		  self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

		gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
		  self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

		num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
		min_num = np.min(num_imgs_per_tracklet)
		max_num = np.max(num_imgs_per_tracklet)
		avg_num = np.mean(num_imgs_per_tracklet)

		num_total_pids = num_train_pids + num_query_pids
		num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

		print("=> MARS loaded")
		print("Dataset statistics:")
		print("	 ------------------------------")
		print("	 subset	  | # ids | # tracklets")
		print("	 ------------------------------")
		print("	 train	  | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
		print("	 query	  | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
		print("	 gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
		print("	 ------------------------------")
		print("	 total	  | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
		print("	 number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
		print("	 ------------------------------")

		self.train = train
		self.query = query
		self.gallery = gallery

		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.root):
			raise RuntimeError("'{}' is not available".format(self.root))
		if not osp.exists(self.train_name_path):
			raise RuntimeError("'{}' is not available".format(self.train_name_path))
		if not osp.exists(self.test_name_path):
			raise RuntimeError("'{}' is not available".format(self.test_name_path))
		if not osp.exists(self.track_train_info_path):
			raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
		if not osp.exists(self.track_test_info_path):
			raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
		if not osp.exists(self.query_IDX_path):
			raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

	def _get_names(self, fpath):
		names = []
		with open(fpath, 'r') as f:
			for line in f:
				new_line = line.rstrip()
				names.append(new_line)
		return names

	def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
		assert home_dir in ['bbox_train', 'bbox_test']
		num_tracklets = meta_data.shape[0]
		pid_list = list(set(meta_data[:,2].tolist()))
		num_pids = len(pid_list)

		if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
		tracklets = []
		num_imgs_per_tracklet = []

		for tracklet_idx in range(num_tracklets):
			data = meta_data[tracklet_idx,...]
			start_index, end_index, pid, camid = data
			if pid == -1: continue # junk images are just ignored
			assert 1 <= camid <= 6
			if relabel: pid = pid2label[pid]
			camid -= 1 # index starts from 0
			img_names = names[start_index-1:end_index]

			# make sure image names correspond to the same person
			pnames = [img_name[:4] for img_name in img_names]
			assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

			# make sure all images are captured under the same camera
			camnames = [img_name[5] for img_name in img_names]
			assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

			# append image names with directory information
			img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
			if len(img_paths) >= min_seq_len:
				img_paths = tuple(img_paths)
				tracklets.append((img_paths, pid, camid))
				num_imgs_per_tracklet.append(len(img_paths))

		num_tracklets = len(tracklets)

		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class iLIDSVID(object):
	"""
	iLIDS-VID
	Reference:
	Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
	
	Dataset statistics:
	# identities: 300
	# tracklets: 600
	# cameras: 2
	Args:
		split_id (int): indicates which split to use. There are totally 10 splits.
	"""
	root = './data/ilids-vid'
	dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
	data_dir = osp.join(root, 'i-LIDS-VID')
	split_dir = osp.join(root, 'train-test people splits')
	split_mat_path = osp.join(split_dir, 'train_test_splits_ilidsvid.mat')
	split_path = osp.join(root, 'splits.json')
	cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
	cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')

	def __init__(self, split_id=0):
		self._download_data()
		self._check_before_run()

		self._prepare_split()
		splits = read_json(self.split_path)
		if split_id >= len(splits):
			raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
		split = splits[split_id]
		train_dirs, test_dirs = split['train'], split['test']
		print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

		train, num_train_tracklets, num_train_pids, num_imgs_train = \
		  self._process_data(train_dirs, cam1=True, cam2=True)
		query, num_query_tracklets, num_query_pids, num_imgs_query = \
		  self._process_data(test_dirs, cam1=True, cam2=False)
		gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
		  self._process_data(test_dirs, cam1=False, cam2=True)

		num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
		min_num = np.min(num_imgs_per_tracklet)
		max_num = np.max(num_imgs_per_tracklet)
		avg_num = np.mean(num_imgs_per_tracklet)

		num_total_pids = num_train_pids + num_query_pids
		num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

		print("=> iLIDS-VID loaded")
		print("Dataset statistics:")
		print("	 ------------------------------")
		print("	 subset	  | # ids | # tracklets")
		print("	 ------------------------------")
		print("	 train	  | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
		print("	 query	  | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
		print("	 gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
		print("	 ------------------------------")
		print("	 total	  | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
		print("	 number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
		print("	 ------------------------------")

		self.train = train
		self.query = query
		self.gallery = gallery

		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids

	def _download_data(self):
		if osp.exists(self.root):
			print("This dataset has been downloaded.")
			return

		mkdir_if_missing(self.root)
		fpath = osp.join(self.root, osp.basename(self.dataset_url))

		print("Downloading iLIDS-VID dataset")
		url_opener = urllib.URLopener()
		url_opener.retrieve(self.dataset_url, fpath)

		print("Extracting files")
		tar = tarfile.open(fpath)
		tar.extractall(path=self.root)
		tar.close()

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.root):
			raise RuntimeError("'{}' is not available".format(self.root))
		if not osp.exists(self.data_dir):
			raise RuntimeError("'{}' is not available".format(self.data_dir))
		if not osp.exists(self.split_dir):
			raise RuntimeError("'{}' is not available".format(self.split_dir))

	def _prepare_split(self):
		if not osp.exists(self.split_path):
			print("Creating splits")
			mat_split_data = loadmat(self.split_mat_path)['ls_set']
			
			num_splits = mat_split_data.shape[0]
			num_total_ids = mat_split_data.shape[1]
			assert num_splits == 10
			assert num_total_ids == 300
			num_ids_each = num_total_ids/2

			# pids in mat_split_data are indices, so we need to transform them
			# to real pids
			person_cam1_dirs = os.listdir(self.cam_1_path)
			person_cam2_dirs = os.listdir(self.cam_2_path)

			# make sure persons in one camera view can be found in the other camera view
			assert set(person_cam1_dirs) == set(person_cam2_dirs)

			splits = []
			for i_split in range(num_splits):
				# first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
				train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
				test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
				
				train_idxs = [int(i)-1 for i in train_idxs]
				test_idxs = [int(i)-1 for i in test_idxs]
				
				# transform pids to person dir names
				train_dirs = [person_cam1_dirs[i] for i in train_idxs]
				test_dirs = [person_cam1_dirs[i] for i in test_idxs]
				
				split = {'train': train_dirs, 'test': test_dirs}
				splits.append(split)

			print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
			print("Split file is saved to {}".format(self.split_path))
			write_json(splits, self.split_path)

		print("Splits created")

	def _process_data(self, dirnames, cam1=True, cam2=True):
		tracklets = []
		num_imgs_per_tracklet = []
		dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
		
		for dirname in dirnames:
			if cam1:
				person_dir = osp.join(self.cam_1_path, dirname)
				img_names = glob.glob(osp.join(person_dir, '*.png'))
				assert len(img_names) > 0
				img_names = tuple(img_names)
				pid = dirname2pid[dirname]
				tracklets.append((img_names, pid, 0))
				num_imgs_per_tracklet.append(len(img_names))

			if cam2:
				person_dir = osp.join(self.cam_2_path, dirname)
				img_names = glob.glob(osp.join(person_dir, '*.png'))
				assert len(img_names) > 0
				img_names = tuple(img_names)
				pid = dirname2pid[dirname]
				tracklets.append((img_names, pid, 1))
				num_imgs_per_tracklet.append(len(img_names))

		num_tracklets = len(tracklets)
		num_pids = len(dirnames)

		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class PRID(object):
	"""
	PRID
	Reference:
	Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
	
	Dataset statistics:
	# identities: 200
	# tracklets: 400
	# cameras: 2
	Args:
		split_id (int): indicates which split to use. There are totally 10 splits.
		min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
	"""
	root = './data/prid2011'
	dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
	split_path = osp.join(root, 'splits_prid2011.json')
	cam_a_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_a')
	cam_b_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_b')

	def __init__(self, split_id=0, min_seq_len=0):
		self._check_before_run()
		splits = read_json(self.split_path)
		if split_id >=	len(splits):
			raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
		split = splits[split_id]
		train_dirs, test_dirs = split['train'], split['test']
		print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

		train, num_train_tracklets, num_train_pids, num_imgs_train = \
		  self._process_data(train_dirs, cam1=True, cam2=True)
		query, num_query_tracklets, num_query_pids, num_imgs_query = \
		  self._process_data(test_dirs, cam1=True, cam2=False)
		gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
		  self._process_data(test_dirs, cam1=False, cam2=True)

		num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
		min_num = np.min(num_imgs_per_tracklet)
		max_num = np.max(num_imgs_per_tracklet)
		avg_num = np.mean(num_imgs_per_tracklet)

		num_total_pids = num_train_pids + num_query_pids
		num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

		print("=> PRID-2011 loaded")
		print("Dataset statistics:")
		print("	 ------------------------------")
		print("	 subset	  | # ids | # tracklets")
		print("	 ------------------------------")
		print("	 train	  | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
		print("	 query	  | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
		print("	 gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
		print("	 ------------------------------")
		print("	 total	  | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
		print("	 number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
		print("	 ------------------------------")

		self.train = train
		self.query = query
		self.gallery = gallery

		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.root):
			raise RuntimeError("'{}' is not available".format(self.root))

	def _process_data(self, dirnames, cam1=True, cam2=True):
		tracklets = []
		num_imgs_per_tracklet = []
		dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
		
		for dirname in dirnames:
			if cam1:
				person_dir = osp.join(self.cam_a_path, dirname)
				img_names = glob.glob(osp.join(person_dir, '*.png'))
				assert len(img_names) > 0
				img_names = tuple(img_names)
				pid = dirname2pid[dirname]
				tracklets.append((img_names, pid, 0))
				num_imgs_per_tracklet.append(len(img_names))

			if cam2:
				person_dir = osp.join(self.cam_b_path, dirname)
				img_names = glob.glob(osp.join(person_dir, '*.png'))
				assert len(img_names) > 0
				img_names = tuple(img_names)
				pid = dirname2pid[dirname]
				tracklets.append((img_names, pid, 1))
				num_imgs_per_tracklet.append(len(img_names))

		num_tracklets = len(tracklets)
		num_pids = len(dirnames)

		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class Dataset(object):
	#deve restituire tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
	#tracklets: lista di clip 
	#num_tracklets: num di clip 
	#num_pids: numero di persone(?)
	#num_imgs_per_tracklet: numero di immagini nel tracklet

	def __init__(self, min_seq_len=0, data_path="./data", id_train="1-21", id_test="101-121", unbalanced=False, seq_len=5):
		print("Loading from {} ...".format(data_path))
		print("- unbalanced={} - seq_len={}".format(unbalanced,seq_len))
		# root = "D:/Downloads/TVPR2 -150-Orig"
		self.root=data_path
		self.unbalanced=unbalanced
		self.seq_len=seq_len
		self.train_dir = osp.join(self.root, 'train/')
		self.test_dir = osp.join(self.root, 'test/')
		self.depth_dir_train = osp.join(self.root, 'npy-train/')
		self.depth_dir_test = osp.join(self.root, 'npy-test/')

		self.lista_file_train = glob.glob(self.train_dir + '/*')  # prendo tutti i file
		self.lista_file_test = glob.glob(self.test_dir + '/*')

		print("# train frames: {}, # test frames {}".format(len(self.lista_file_train), len(self.lista_file_test)))

		#Recupero gli IDs
		ids_train = id_train.split("-")
		ids_test = id_test.split("-")
		self.id_train_init = int(ids_train[0])
		self.id_train_end = int(ids_train[1])
		self.id_test_init = int(ids_test[0])
		self.id_test_end = int(ids_test[1])

		train, num_train_tracklets, num_train_pids, num_imgs_train = self._process_data_general("Train") 			# self._process_data1(self.train_dir)
		query, num_query_tracklets, num_query_pids, num_imgs_query = self._process_data_general("Query") 			# self._process_data2(self.test_dir)
		gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = self._process_data_general("Gallery") 	# self._process_data3(self.test_dir)
		num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
		min_num = np.min(num_imgs_per_tracklet)
		max_num = np.max(num_imgs_per_tracklet)
		avg_num = np.mean(num_imgs_per_tracklet)

		num_total_pids = num_train_pids + num_query_pids
		num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

		print("=> Dataset caricato")
		print("Dataset statistics:")
		print("	 ------------------------------")
		print("	 subset	  | # ids | # tracklets")
		print("	 ------------------------------")
		print("	 train	  | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
		print("	 query	  | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
		print("	 gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
		print("	 ------------------------------")
		print("	 total	  | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
		print("	 number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
		print("	 ------------------------------")

		self.train = train
		self.query = query
		self.gallery = gallery

		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids



			
	def _process_data_general(self, tipo):
		tracklets = []
		num_imgs_per_tracklet = []
		num_pids=0
		if tipo=="Train":
			folder=self.train_dir
			inizio = self.id_train_init
			fine = self.id_train_end
			cam_id=1
		elif tipo=="Gallery":
			folder = self.train_dir
			inizio = self.id_test_init
			fine = self.id_test_end
			cam_id = 1
		elif tipo=="Query":
			folder=self.test_dir
			inizio = self.id_test_init
			fine = self.id_test_end
			cam_id = 2

		print("{}...".format(tipo))
		for i in range(inizio,fine):
			stringa='Image-'+str(i)+'-*.jpg'
			stringa_depth='ImageDepth-'+str(i)+'-*.npy'
			clip=glob.glob(folder+stringa) #raccolgo tutti i frame con pid=i
			clip_depth=glob.glob(self.depth_dir_train+stringa_depth) #raccolgo tutte le relative immagini depth

			#DEVO AVERE LO STESSO NUMERO DI FRAME RGB e DEPTH
			minimo=min(len(clip),len(clip_depth))
			clip = clip[:minimo]
			clip_depth = clip_depth[:minimo]

			if (len(clip))!=0:
				if self.unbalanced:
					for j in range(0,minimo,self.seq_len):
						jfine=j+self.seq_len
						tracklets.append((clip[j:jfine], clip_depth[j:jfine], num_pids, cam_id))
						num_imgs_per_tracklet.append(len(clip))
					num_pids += 1

				else:
					pid = i - inizio
					tracklets.append((clip,clip_depth,num_pids,cam_id))
					num_pids+=1
					num_imgs_per_tracklet.append(len(clip))

		num_tracklets= len(tracklets)
		print("---> ids {} to {} - founded {} pids - {} frames.".format(inizio, fine, num_pids,sum(num_imgs_per_tracklet)))
		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
			
	def _process_data1(self, train_dir):
		tracklets = []
		num_imgs_per_tracklet = []
		num_pids=0
		inizio=self.id_train_init
		fine=self.id_train_end
		for i in range(inizio,fine):
			stringa='Image-'+str(i)+'-*.jpg'
			stringa_depth='ImageDepth-'+str(i)+'-*.npy'
			clip=glob.glob(train_dir+stringa) #raccolgo tutti i frame con pid=i
			clip_depth=glob.glob(self.depth_dir_train+stringa_depth) #raccolgo tutte le relative immagini depth
			#for frame in listafile:
				#if stringa in frame:
					#clip.append(frame)
			if (len(clip))!=0:
				#clip=tuple(clip)
				pid = i - inizio
				tracklets.append((clip,clip_depth,num_pids,1))
				#print(tracklets)
				num_pids+=1
				num_imgs_per_tracklet.append(len(clip))
				#del clip[:]
		num_tracklets= len(tracklets)
		print("Train: ids {} to {} - founded {} pids - {} frames.".format(self.id_train_init, self.id_train_end, num_pids,sum(num_imgs_per_tracklet)))
		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
		
		
	def _process_data2(self, test_dir):
		tracklets = []
		num_imgs_per_tracklet = []
		num_pids=0
		inizio=self.id_test_init
		fine=self.id_test_end
		for i in range(inizio,fine):
			stringa='Image-'+str(i)+'-*.jpg'
			stringa_depth='ImageDepth-'+str(i)+'-*.npy'
			clip=glob.glob(test_dir+stringa) #raccolgo tutti i frame con pid=i
			clip_depth=glob.glob(self.depth_dir_test+stringa_depth) #raccolgo tutte le relative immagini depth
			#for frame in listafile:
				#if stringa in frame:
					#clip.append(frame)
			if (len(clip))!=0:
				#clip=tuple(clip)
				pid = i - inizio
				limite = int(len(clip) * 0.7)
				limite2 = int(len(clip_depth) * 0.7)
				clip2 = clip[limite:]
				clip_depth2 = clip_depth[limite2:]
				tracklets.append((clip2, clip_depth2, num_pids, 2))
				num_pids+=1
				num_imgs_per_tracklet.append(len(clip2))
				#del clip[:]
		num_tracklets= len(tracklets)
		print("Query: ids {} to {} - founded {} pids - {} frames.".format(self.id_test_init, self.id_test_end, num_pids,sum(num_imgs_per_tracklet)))
		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
		
	def _process_data3(self, test_dir):
		tracklets = []
		num_imgs_per_tracklet = []
		num_pids=0
		inizio=self.id_test_init
		fine=self.id_test_end
		for i in range(inizio,fine):
			stringa='Image-'+str(i)+'-*.jpg'
			stringa_depth='ImageDepth-'+str(i)+'-*.npy'
			clip=glob.glob(test_dir+stringa) #raccolgo tutti i frame con pid=i
			clip_depth=glob.glob(self.depth_dir_test+stringa_depth) #raccolgo tutte le relative immagini depth
			#for frame in listafile:
				#if stringa in frame:
					#clip.append(frame)
			if (len(clip))!=0:
				#clip=tuple(clip)
				pid = i - inizio
				limite = int(len(clip) * 0.7)
				limite2 = int(len(clip_depth) * 0.7)
				clip2 = clip[:limite]
				clip_depth2 = clip_depth[:limite2]
				tracklets.append((clip2, clip_depth2, num_pids, 1))
				#print(tracklets)
				num_pids+=1
				num_imgs_per_tracklet.append(len(clip2))
				#del clip[:]
		num_tracklets= len(tracklets)
		print("Gallery: ids {} to {} - founded {} pids - {} frames.".format(self.id_test_init, self.id_test_end, num_pids,sum(num_imgs_per_tracklet)))
		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
		
		
	def _process_data(self, tipo):

		num_pids = 0
		num_tracklets = 0
		tracklets = []
		num_imgs_per_tracklet = []


		print(tipo)
		if tipo == "train":
			camid = 1
			inizio=1
			fine=500
			for i in range(inizio, fine+1):
				paths = glob.glob(self.root + "/Train100/" + "Image-{}-*".format(i))
				if len(paths) == 0:
					#print("ID={} skipped".format(i))
					continue
				num_pids += 1
				num_tracklets += 1
				img_paths = []
				for j in range(200):
					p = self.root + "/Train100/" + "Image-{}-{}.jpg".format(i, j)
					if os.path.exists(p):
						img_paths.append(p)

				pid = i - inizio
				tracklets.append((img_paths, pid, camid))
				num_imgs_per_tracklet.append(len(img_paths))
				#print("{}) {} images".format(i, len(paths)))

		elif tipo == "gallery":
			camid = 1
			inizio = 1000
			fine = 1100
			for i in range(inizio, fine+1):
				paths = glob.glob(self.root + "/Test100/" + "Image-{}-*".format(i))
				if len(paths) == 0:
					#print("ID={} skipped".format(i))
					continue
				num_pids += 1
				num_tracklets += 1
				img_paths = []
				for j in range(200):
					p = self.root + "/Test100/" + "Image-{}-{}.jpg".format(i, j)
					if os.path.exists(p):
						img_paths.append(p)

				pid = i - inizio
				limite = int(len(img_paths) * 0.7)
				img_paths2 = img_paths[:limite]
				tracklets.append((img_paths2, pid, camid))
				num_imgs_per_tracklet.append(len(img_paths2))
				#print("{}) {} images".format(i, len(img_paths2)))

		elif tipo == "query":
			camid = 2
			inizio = 1000
			fine = 1100
			for i in range(inizio, fine+1):
				paths = glob.glob(self.root + "/Test100/" + "Image-{}-*".format(i))
				if len(paths) == 0:
					#print("ID={} skipped".format(i))
					continue
				num_pids += 1
				num_tracklets += 1
				img_paths = []
				for j in range(200):
					p = self.root + "/Test100/" + "Image-{}-{}.jpg".format(i, j)
					if os.path.exists(p):
						img_paths.append(p)

				pid = i - inizio
				limite = int(len(img_paths) * 0.7)
				img_paths2 = img_paths[limite:]
				tracklets.append((img_paths2, pid, camid))
				num_imgs_per_tracklet.append(len(img_paths2))
				#print("{}) {} images".format(i, len(img_paths2)))

		print("\nSTATS: {} traklets - {} pids\n".format(num_tracklets, num_pids))

		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class Rocco(object):
	# deve restituire tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
	# tracklets: lista di clip
	# num_tracklets: num di clip
	# num_pids: numero di persone(?)
	# num_imgs_per_tracklet: numero di immagini nel tracklet

	def __init__(self, min_seq_len=0, data_path="./data", id_train=None, id_test=None,  seq_len=5):
		#print("Loading from {} ...".format(data_path))
		#print("- seq_len={}".format(seq_len))
		# root = "D:/Downloads/TVPR2 -150-Orig"
		self.root = data_path
		self.seq_len = seq_len


		train, num_train_tracklets, num_train_pids, num_imgs_train = self._process_data("Train",id_train)
		query, num_query_tracklets, num_query_pids, num_imgs_query = self._process_data("Query",id_test)
		gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = self._process_data("Gallery",id_train)
		num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
		min_num = np.min(num_imgs_per_tracklet)
		max_num = np.max(num_imgs_per_tracklet)
		avg_num = np.mean(num_imgs_per_tracklet)

		num_total_pids = num_train_pids + num_query_pids
		num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

		'''
		print("=> Dataset caricato")
		print("Dataset statistics:")
		print("	 ------------------------------")
		print("	 subset	  | # ids | # tracklets")
		print("	 ------------------------------")
		print("	 train	  | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
		print("	 query	  | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
		print("	 gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
		print("	 ------------------------------")
		print("	 total	  | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
		print("	 number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
		print("	 ------------------------------")
		'''

		self.train = train
		self.query = query
		self.gallery = gallery

		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids

	def _process_data_general(self, tipo):
		tracklets = []
		num_imgs_per_tracklet = []
		num_pids = 0
		if tipo == "Train":
			folder = self.train_dir
			inizio = self.id_train_init
			fine = self.id_train_end
			cam_id = 1
		elif tipo == "Gallery":
			folder = self.train_dir
			inizio = self.id_test_init
			fine = self.id_test_end
			cam_id = 1
		elif tipo == "Query":
			folder = self.test_dir
			inizio = self.id_test_init
			fine = self.id_test_end
			cam_id = 2

		print("{}...".format(tipo))
		for i in range(inizio, fine):
			stringa = 'Image-' + str(i) + '-*.jpg'
			stringa_depth = 'ImageDepth-' + str(i) + '-*.npy'
			clip = glob.glob(folder + stringa)  # raccolgo tutti i frame con pid=i
			clip_depth = glob.glob(self.depth_dir_train + stringa_depth)  # raccolgo tutte le relative immagini depth

			# DEVO AVERE LO STESSO NUMERO DI FRAME RGB e DEPTH
			minimo = min(len(clip), len(clip_depth))
			clip = clip[:minimo]
			clip_depth = clip_depth[:minimo]

			if (len(clip)) != 0:
				if self.unbalanced:
					for j in range(0, minimo, self.seq_len):
						jfine = j + self.seq_len
						tracklets.append((clip[j:jfine], clip_depth[j:jfine], num_pids, cam_id))
						num_imgs_per_tracklet.append(len(clip))
					num_pids += 1

				else:
					pid = i - inizio
					tracklets.append((clip, clip_depth, num_pids, cam_id))
					num_pids += 1
					num_imgs_per_tracklet.append(len(clip))

		num_tracklets = len(tracklets)
		print("---> ids {} to {} - founded {} pids - {} frames.".format(inizio, fine, num_pids,	sum(num_imgs_per_tracklet)))
		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

	def _process_data(self, tipo, lines):
		tracklets = []
		num_imgs_per_tracklet = []
		num_pids = 0
		if tipo == "Train":
			#folder = self.train_dir
			#inizio = self.id_train_init
			#fine = self.id_train_end
			cam_id = 1
			self.mapid_train={}
			mapid = self.mapid_train
		elif tipo == "Gallery":
			#folder = self.train_dir
			#inizio = self.id_test_init
			#fine = self.id_test_end
			cam_id = 1
			self.mapid_gallery = {}
			mapid = self.mapid_gallery
		elif tipo == "Query":
			#folder = self.test_dir
			#inizio = self.id_test_init
			#fine = self.id_test_end
			cam_id = 2
			self.mapid_query = {}
			mapid = self.mapid_query

		#print("{}...".format(tipo))

		for l in lines:
			s=l.strip().split(";")
			idc=int(s[0])
			idm=int(s[1])
			imgs=s[2:]
			if tipo == "Query":	self.query_imgs=imgs

			clip = glob.glob("{}{}/*_rgb.png".format(self.root,idc))  # raccolgo tutti i frame con pid=i
			clip_depth = glob.glob("{}{}/*_depth.png".format(self.root,idc))  # raccolgo tutte le relative immagini depth

			if (len(clip)) != 0:
				if idm not in mapid:
					mapid[idm]=num_pids
					tracklets.append((clip, clip_depth, num_pids, cam_id))
					num_pids += 1
					num_imgs_per_tracklet.append(len(clip))
				else:
					clip_old, clip_depth_old, num_pids_old, cam_id_old = tracklets[mapid[idm]]
					tracklets[mapid[idm]] = (clip_old+clip,clip_depth_old+clip_depth, num_pids_old, cam_id_old)
					num_imgs_per_tracklet[mapid[idm]]=num_imgs_per_tracklet[mapid[idm]]+len(clip)

		num_tracklets = len(tracklets)
		#print("---> rows {} - founded {} pids - {} frames.".format(len(lines), num_pids, sum(num_imgs_per_tracklet)))
		if tipo in ["Gallery","Query"]:
			print("{} ---> mapping:{}".format(tipo,mapid))
			for i in mapid:
				print("-- {} - {} - {} frames".format(mapid[i],i,len(tracklets[mapid[i]][0])))
		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


				
"""Create dataset"""

__factory = {
	#'mars': Mars,
	#'ilidsvid': iLIDSVID,
	#'prid': PRID,
	'dataset': Dataset,
	'rocco' : Rocco,
}

def get_names():
	return __factory.keys()

def init_dataset(name, *args, **kwargs):
	if name not in __factory.keys():
		raise KeyError("Unknown dataset: {}".format(name))
	return __factory[name](*args, **kwargs)

if __name__ == '__main__':
	# test
	#dataset = Market1501()
	#dataset = Mars()
	dataset = iLIDSVID()
	dataset = PRID()
	dataset = Dataset()






