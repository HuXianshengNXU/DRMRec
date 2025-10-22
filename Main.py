import time

import torch
from torch import nn
import Utils.TimeLogger as logger
from MADR import MADR
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix
from torch.nn import MSELoss

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG', 'Precision']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
		self.val_mat = self.handler.loadOneFile(self.handler.predir + 'valMat.pkl').tocsr()
		self.rl_initialized = False

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')

		recallMax = 0
		ndcgMax = 0
		precisionMax = 0
		bestEpoch = 0

		log('Model Initialized')

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			if ep < args.pretrain_epochs:
				reses = self.trainEpoch(diffusion_only=True)
			else:
				reses = self.trainEpoch(diffusion_only=False)

			log(self.makePrint('Train', ep, reses, tstFlag))

			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()

		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision',
			  precisionMax)

	def prepareModel(self):
		if args.data == 'tiktok':
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(),
							   self.handler.audio_feats.detach()).cuda()
		else:
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()

		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
			out_dims = eval(args.dims) + [args.item]
			in_dims = out_dims[::-1]
			self.denoise_model_audio = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
			self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)

		self.ppo_image = MADR(self.denoise_model_image, lr=args.rl_lr, gamma=args.gamma, eps_clip=args.eps_clip)
		self.ppo_text = MADR(self.denoise_model_text, lr=args.rl_lr, gamma=args.gamma, eps_clip=args.eps_clip)
		if args.data == 'tiktok':
			self.ppo_audio = MADR(self.denoise_model_audio, lr=args.rl_lr, gamma=args.gamma, eps_clip=args.eps_clip)
		self.rl_initialized = True

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def calc_reward(self, usr_embeds, itm_embeds, topk=20):
		rewards = []
		val_users = [u for u in range(self.val_mat.shape[0]) if len(self.val_mat[u].indices) > 0]
		if not val_users:
			return torch.tensor(0.0, dtype=torch.float32).cuda()

		val_usr_embeds = usr_embeds[val_users]

		scores = torch.mm(val_usr_embeds, itm_embeds.t())

		for i, u in enumerate(val_users):
			true_items = set(self.val_mat[u].indices)

			user_scores = scores[i]

			_, top_20_indices = torch.topk(user_scores, 20)
			_, top_10_indices = torch.topk(user_scores, 10)
			_, top_50_indices = torch.topk(user_scores, 50)

			top_items_20 = set(top_20_indices.cpu().numpy())
			top_items_10 = set(top_10_indices.cpu().numpy())
			top_items_50 = set(top_50_indices.cpu().numpy())

			recall_20 = len(true_items & top_items_20) / len(true_items)
			precision_10 = len(true_items & top_items_10) / 10
			ndcg = self.calc_ndcg(top_items_20, true_items)

			reward = 0.5 * recall_20 + 0.3 * precision_10 + 0.2 * ndcg
			rewards.append(reward)

		return torch.tensor(rewards, dtype=torch.float32).mean().cuda()

	def calc_ndcg(self, top_items, true_items):
		dcg = 0.0
		idcg = sum(1.0 / np.log2(i + 2) for i in range(len(true_items)))
		if idcg == 0:
			return 0.0
		for i, item in enumerate(top_items):
			if item in true_items:
				dcg += 1.0 / np.log2(i + 2)
		return dcg / idcg

	def trainEpoch(self, diffusion_only=True):
		self.model.train()
		self.denoise_model_image.train()
		self.denoise_model_text.train()
		if args.data == 'tiktok':
			self.denoise_model_audio.train()

		lossSum = 0.0
		batchCnt = 0


		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		# epLoss, epRecLoss = 0, 0
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss_image, epDiLoss_text = 0, 0
		if args.data == 'tiktok':
			epDiLoss_audio = 0
		diffusionLoader = self.handler.diffusionLoader

		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			if args.data == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()

			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()
			if args.data == 'tiktok':
				self.denoise_opt_audio.zero_grad()


			if args.data == 'tiktok':
				diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(
					self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, cross_feats=[text_feats, audio_feats])
				diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(
					self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats, cross_feats=[image_feats, audio_feats])
				diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(
					self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats, cross_feats=[image_feats, text_feats])

			else:
				diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(
					self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, cross_feats=[text_feats])
				diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(
					self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats, cross_feats=[image_feats])

			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
			if args.data == 'tiktok':
				loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss

			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()
			if args.data == 'tiktok':
				epDiLoss_audio += loss_audio.item()

			if args.data == 'tiktok':
				loss = loss_image + loss_text + loss_audio
			else:
				loss = loss_image + loss_text

			loss.backward()
			self.denoise_opt_image.step()
			self.denoise_opt_text.step()
			if args.data == 'tiktok':
				self.denoise_opt_audio.step()

			log(f'Diffusion Step {i}/{len(diffusionLoader)}', save=False, oneline=True)
		log('')

		log('Start to re-build UI matrix')
		with torch.no_grad():
			u_list_image, i_list_image, edge_list_image = [], [], []
			u_list_text, i_list_text, edge_list_text = [], [], []
			if args.data == 'tiktok':
				u_list_audio, i_list_audio, edge_list_audio = [], [], []

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				denoised_batch = self.diffusion_model.p_sample(
					self.denoise_model_image, batch_item, args.sampling_steps, args.sampling_noise) + args.weight_diff_res * batch_item
				_, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]):
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				denoised_batch = self.diffusion_model.p_sample(
					self.denoise_model_text, batch_item, args.sampling_steps, args.sampling_noise) + args.weight_diff_res * batch_item
				_, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]):
						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if args.data == 'tiktok':
					denoised_batch = self.diffusion_model.p_sample(
						self.denoise_model_audio, batch_item, args.sampling_steps, args.sampling_noise) + args.weight_diff_res * batch_item
					_, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
					for i in range(batch_index.shape[0]):
						for j in range(indices_[i].shape[0]):
							u_list_audio.append(int(batch_index[i].cpu().numpy()))
							i_list_audio.append(int(indices_[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)

			self.image_UI_matrix = self.buildUIMatrix(
				np.array(u_list_image), np.array(i_list_image), np.array(edge_list_image))
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			self.text_UI_matrix = self.buildUIMatrix(
				np.array(u_list_text), np.array(i_list_text), np.array(edge_list_text))
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			if args.data == 'tiktok':
				self.audio_UI_matrix = self.buildUIMatrix(
					np.array(u_list_audio), np.array(i_list_audio), np.array(edge_list_audio))
				self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)
		log('UI matrix built!')

		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			if args.data == 'tiktok':
				usrEmbeds, itmEmbeds,usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix,
															 self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds, itmEmbeds,usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix,
															 self.text_UI_matrix)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = self.model.reg_loss() * args.reg
			loss = bprLoss + regLoss

			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			if args.cl_method == 1:
				clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
				clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
				if args.data == 'tiktok':
					clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
					clLoss_ = clLoss1 + clLoss2 + clLoss3
				else:
					clLoss_ = clLoss1 + clLoss2
			else :
				if args.data == 'tiktok':
					clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
					clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
					clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				else:
					clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			if args.cl_method == 1:
				clLoss = clLoss_

			loss += clLoss

			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()

			log(f'Rec Step {i}/{len(trnLoader)}', save=False, oneline=True)
		log('')

		if not diffusion_only and self.rl_initialized and usrEmbeds is not None and itmEmbeds is not None:
			log('Starting RL fine-tuning...')
			reward = self.calc_reward(usrEmbeds, itmEmbeds, topk=args.topk)
			log(f'RL Reward: {reward.item():.4f}')

			states_image, actions_image, old_log_probs_image, timesteps_image = [], [], [], []
			states_text, actions_text, old_log_probs_text, timesteps_text = [], [], [], []
			if args.data == 'tiktok':
				states_audio, actions_audio, old_log_probs_audio, timesteps_audio = [], [], [], []

			for batch in diffusionLoader:
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
				timesteps = torch.randint(0, args.steps, (batch_item.shape[0],)).cuda()

				noise_image = torch.randn_like(batch_item) * args.noise_scale
				state_image = batch_item + noise_image
				action_image = self.denoise_model_image(state_image, timesteps) + args.weight_rl_res * batch_item
				log_prob_image = self.denoise_model_image.get_log_prob(state_image, timesteps, action_image)
				states_image.append(state_image)
				actions_image.append(action_image)
				old_log_probs_image.append(log_prob_image)
				timesteps_image.append(timesteps)

				noise_text = torch.randn_like(batch_item) * args.noise_scale
				state_text = batch_item + noise_text
				action_text = self.denoise_model_text(state_text, timesteps) + args.weight_rl_res * batch_item
				log_prob_text = self.denoise_model_text.get_log_prob(state_text, timesteps, action_text)
				states_text.append(state_text)
				actions_text.append(action_text)
				old_log_probs_text.append(log_prob_text)
				timesteps_text.append(timesteps)

				if args.data == 'tiktok':
					noise_audio = torch.randn_like(batch_item) * args.noise_scale
					state_audio = batch_item + noise_audio
					action_audio = self.denoise_model_audio(state_audio, timesteps) + args.weight_rl_res * batch_item
					log_prob_audio = self.denoise_model_audio.get_log_prob(state_audio, timesteps, action_audio)
					states_audio.append(state_audio)
					actions_audio.append(action_audio)
					old_log_probs_audio.append(log_prob_audio)
					timesteps_audio.append(timesteps)

			states_image_cat = torch.cat(states_image)
			actions_image_cat = torch.cat(actions_image)
			old_log_probs_image_cat = torch.cat(old_log_probs_image)
			timesteps_image_cat = torch.cat(timesteps_image)

			if states_image_cat.numel() == 0:
				log('Warning: Empty states for image modality, skipping PPO update')
			else:
				values_image = torch.zeros_like(states_image_cat[:, 0], device=states_image_cat.device)

				rewards_image = reward.repeat(len(values_image))

				self.ppo_image.update(
					states_image_cat,
					actions_image_cat,
					old_log_probs_image_cat,
					rewards_image,
					values_image,
					timesteps_image_cat,
					epochs=args.ppo_epochs
				)

			states_text_cat = torch.cat(states_text)
			actions_text_cat = torch.cat(actions_text)
			old_log_probs_text_cat = torch.cat(old_log_probs_text)
			timesteps_text_cat = torch.cat(timesteps_text)

			if states_text_cat.numel() == 0:
				log('Warning: Empty states for text modality, skipping PPO update')
			else:
				values_text = torch.zeros_like(states_text_cat[:, 0], device=states_text_cat.device)

				rewards_text = reward.repeat(len(values_text))
				self.ppo_text.update(
					states_text_cat,
					actions_text_cat,
					old_log_probs_text_cat,
					rewards_text,
					values_text,
					timesteps_text_cat,
					epochs=args.ppo_epochs
				)

			if args.data == 'tiktok':
				states_audio_cat = torch.cat(states_audio)
				actions_audio_cat = torch.cat(actions_audio)
				old_log_probs_audio_cat = torch.cat(old_log_probs_audio)
				timesteps_audio_cat = torch.cat(timesteps_audio)

				if states_audio_cat.numel() == 0:
					log('Warning: Empty states for audio modality, skipping PPO update')
				else:
					values_audio = torch.zeros_like(states_audio_cat[:, 0], device=states_audio_cat.device)

					rewards_audio = reward.repeat(len(values_audio))
					self.ppo_audio.update(
						states_audio_cat,
						actions_audio_cat,
						old_log_probs_audio_cat,
						rewards_audio,
						values_audio,
						timesteps_audio_cat,
						epochs=args.ppo_epochs
					)
			log('RL fine-tuning completed')

		ret = dict()
		ret['Loss'] = epLoss / len(trnLoader) if len(trnLoader) > 0 else 0
		ret['BPR Loss'] = epRecLoss / len(trnLoader) if len(trnLoader) > 0 else 0
		ret['Di image loss'] = epDiLoss_image / len(diffusionLoader) if len(diffusionLoader) > 0 else 0
		ret['Di text loss'] = epDiLoss_text / len(diffusionLoader) if len(diffusionLoader) > 0 else 0
		if args.data == 'tiktok':
			ret['Di audio loss'] = epDiLoss_audio / len(diffusionLoader) if len(diffusionLoader) > 0 else 0
		return ret

	def testEpoch(self):
		self.model.eval()
		trnLoader = self.handler.trnLoader
		tstLoader = self.handler.tstLoader
		recall, ndcg, precision = 0, 0, 0
		num = 0

		with torch.no_grad():
			if args.data == 'tiktok':
				usrEmbeds, itmEmbeds, usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_MM(
					self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds, itmEmbeds, usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(
					self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

			for usr, trnMask in tstLoader:
				usr = usr.long().cuda()
				trnMask = trnMask.cuda()
				batchsize = usr.shape[0]
				num += batchsize
				uEmbeds = usrEmbeds[usr]
				scores = torch.mm(uEmbeds, itmEmbeds.t())
				scores = scores - 1e8 * trnMask

				_, topLocs = torch.topk(scores, args.topk)
				tstLocs = self.handler.tstLoader.dataset.tstLocs
				for i in range(batchsize):
					temRecall, temNdcg, temPrecision = calcMetrics(
						topLocs[i].cpu().numpy(), tstLocs[usr[i].item()])
					recall += temRecall
					ndcg += temNdcg
					precision += temPrecision

		self.model.train()
		recall /= num
		ndcg /= num
		precision /= num
		return {'Recall': recall, 'NDCG': ndcg, 'Precision': precision}

def calcMetrics(topLocs, tstLocs):
	tstSet = set(tstLocs)
	if len(tstSet) == 0:
		return 0, 0, 0
	recall = len(set(topLocs) & tstSet) / len(tstSet)
	precision = len(set(topLocs) & tstSet) / len(topLocs) if len(topLocs) > 0 else 0
	dcg = 0
	idcg = np.sum(1.0 / np.log2(np.arange(2, len(tstSet) + 2)))
	for i, loc in enumerate(topLocs):
		if loc in tstSet:
			dcg += 1.0 / np.log2(i + 2)
	ndcg = dcg / idcg if idcg != 0 else 0
	return recall, ndcg, precision

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)


if __name__ == '__main__':
	print(args)
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()

	print(args)


