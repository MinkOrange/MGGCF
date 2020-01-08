import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NCF(nn.Module):
	def __init__(self, user_num, item_num, group_num, factor_num, num_layers,dropout):
		super(NCF, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		group_num: number of groups;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		
		"""
		self.dropout = dropout

		self.embed_user_GMF = nn.Embedding(user_num, factor_num)
		self.embed_item_GMF = nn.Embedding(item_num, factor_num)
		self.embed_group_GMF = nn.Embedding(group_num,factor_num)
		self.embed_user_MLP = nn.Embedding(
				user_num, factor_num * (2 ** (num_layers - 1)))
		self.embed_item_MLP = nn.Embedding(
				item_num, factor_num * (2 ** (num_layers - 1)))
		self.embed_group_MLP = nn.Embedding(
			    group_num, factor_num * (2 ** (num_layers - 1)))

		MLP_modules = []
		for i in range(num_layers):
			input_size = factor_num * (2 ** (num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)


		predict_size = factor_num * 2
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """

		nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
		nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
		nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
		nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
		nn.init.normal_(self.embed_group_GMF.weight, std=0.01)
		nn.init.normal_(self.embed_group_MLP.weight, std=0.01)


		for m in self.MLP_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_uniform_(self.predict_layer.weight,
								a=1, nonlinearity='sigmoid')

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()


	def forward(self, group, user, item):
		if (group is not None) and (user is None):
			out = self.grp_forward(group,item)
		else:
			out = self.usr_forward(user,item)
		return out

	def grp_forward(self,group,item):

		embed_user_GMF = self.embed_user_GMF(group)
		embed_item_GMF = self.embed_item_GMF(item)
		output_GMF = embed_user_GMF * embed_item_GMF

		embed_user_MLP = self.embed_user_MLP(group)
		embed_item_MLP = self.embed_item_MLP(item)
		interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
		output_MLP = self.MLP_layers(interaction)

		concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return prediction.view(-1)
	def usr_forward(self,user,item):
		embed_user_GMF = self.embed_user_GMF(user)
		embed_item_GMF = self.embed_item_GMF(item)
		output_GMF = embed_user_GMF * embed_item_GMF

		embed_user_MLP = self.embed_user_MLP(user)
		embed_item_MLP = self.embed_item_MLP(item)
		interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
		output_MLP = self.MLP_layers(interaction)

		concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return prediction.view(-1)

