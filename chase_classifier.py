import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ResBlock(nn.Module):
	def __init__(self, size=64):
		super(ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(size, size, 3, padding=1)
		self.conv2 = nn.Conv2d(size,size,3,padding = 1)
	def forward(self, x):
		y = F.relu(self.conv1(x))
		y = F.relu(self.conv2(y))
		return x + y

class Chase_Classifier(nn.Module):
	def __init__(self, _supered=False):
		super(Chase_Classifier, self).__init__()
		if _supered:
			return
		# Assumes a 128x128 input
		self.conv1 = nn.Conv2d(3, 64, 6, stride=2)
		self.FPL = nn.MaxPool2d(2)
		self.blocks = nn.ModuleList([ResBlock() for i in range(4)])
		self.blocks.append(nn.Conv2d(64, 128, 3, padding=1))
		self.blocks.extend([ResBlock(size=128) for i in range(3)])
		self.conv2 = nn.Conv2d(128,128, 3, stride=2)
		self.fc = nn.Linear(128*15*15, 10)
		self.last_cnn_size = 128*15*15
	def forward(self, x):
		x = F.relu(self.conv1(x))
		for i in self.blocks:
			x = i(x)
		x = self.FPL(x)
		x = F.relu(self.conv2(x))
		x = x.view(-1,self.last_cnn_size)
		x = self.fc(x)
		return nn.LogSoftmax()(x)
	def get_model_path(self):
		return "./chase_net.pth"
