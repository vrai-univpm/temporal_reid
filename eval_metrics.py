from __future__ import print_function, absolute_import
import numpy as np
import copy

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
	num_q, num_g = distmat.shape
	print(num_q)
	print(num_g)
	if num_g < max_rank:
		max_rank = num_g
		print("Note: number of gallery samples is quite small, got {}".format(num_g))
	indices = np.argsort(distmat, axis=1)
	matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

	# compute cmc curve for each query
	all_cmc = []
	all_AP = []
	num_valid_q = 0.
	for q_idx in range(num_q):
		# get query pid and camid
		q_pid = q_pids[q_idx]
		q_camid = q_camids[q_idx]

		# remove gallery samples that have the same pid and camid with query
		order = indices[q_idx]
		remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
		keep = np.invert(remove)

		# compute cmc curve
		orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
		if not np.any(orig_cmc):
			# this condition is true when query identity does not appear in gallery
			continue

		cmc = orig_cmc.cumsum()
		cmc[cmc > 1] = 1

		all_cmc.append(cmc[:max_rank])
		num_valid_q += 1.

		# compute average precision
		# reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
		num_rel = orig_cmc.sum()
		tmp_cmc = orig_cmc.cumsum()
		tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
		tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
		AP = tmp_cmc.sum() / num_rel
		all_AP.append(AP)


	assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
	all_cmc = np.asarray(all_cmc).astype(np.float32)
	all_cmc = all_cmc.sum(0) / num_valid_q
	mAP = np.mean(all_AP)
	
	return all_cmc, mAP

	return


def evaluate_distance(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
	num_q, num_g = distmat.shape
	#print(num_q)
	#print(num_g)
	'''
	if num_g < max_rank:
		max_rank = num_g
		print("Note: number of gallery samples is quite small, got {}".format(num_g))
	'''
	indices = np.argsort(distmat, axis=1)
	matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

	# compute cmc curve for each query
	all_cmc = []
	all_AP = []
	num_valid_q = 0.
	distances = []
	distances_wrong = []
	for q_idx in range(num_q):
		# get query pid and camid
		q_pid = q_pids[q_idx]
		q_camid = q_camids[q_idx]

		# remove gallery samples that have the same pid and camid with query
		order = indices[q_idx]
		remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
		keep = np.invert(remove)

		#print("order={} - distance={}".format(order,distmat[q_idx, order[0]]))

		'''
		if g_pids[order[0]] == q_pid:
			print("g_pid={} - q_pid={} - distance={} ---> CORRECT!".format(g_pids[order[0]], q_pid, distmat[q_idx, order[0]]))
			distances.append(distmat[q_idx, order[0]])
		else:
			print("g_pid={} - q_pid={} - distance={} ---> WRONG!".format(g_pids[order[0]], q_pid, distmat[q_idx, order[0]]))
			distances_wrong.append(distmat[q_idx, order[0]])
		'''
		return order[0], distmat[q_idx, order[0]]

		'''
		# compute cmc curve
		orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
		if not np.any(orig_cmc):
			# this condition is true when query identity does not appear in gallery
			continue

		cmc = orig_cmc.cumsum()
		cmc[cmc > 1] = 1

		all_cmc.append(cmc[:max_rank])
		num_valid_q += 1.

		# compute average precision
		# reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
		num_rel = orig_cmc.sum()
		tmp_cmc = orig_cmc.cumsum()
		tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
		tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
		AP = tmp_cmc.sum() / num_rel
		all_AP.append(AP)
		'''
	'''
	assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
	print("Distances MIN={} - MAX={}".format(min(distances),max(distances)))
	print("Distances WRONG = {}".format(distances_wrong))

	all_cmc = np.asarray(all_cmc).astype(np.float32)
	all_cmc = all_cmc.sum(0) / num_valid_q
	mAP = np.mean(all_AP)

	return all_cmc, mAP
	'''
	return
