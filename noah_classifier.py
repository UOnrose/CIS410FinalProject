from chase_classifier import Chase_Classifier
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Noah_Classifier(Chase_Classifier):
	def __init__(self):
		'''
		Initializer for Noah_Classifier
		creates linear layers and dropouts
		'''
		# Edit by Chase, added in super argument to prevent calling issues.
		super(Noah_Classifier, self).__init__(_supered=True)

		# Linear Layers
		self.fc1 = nn.Linear(128*128*3, 64) # Add false as third arg if no bias
		self.fc2 = nn.Linear(300,300)
		self.fc3 = nn.Linear(300,300)
		self.fc4 = nn.Linear(300,267)

		# Dropouts
		self.d10 = nn.Dropout2d(0.1)
		self.d25 = nn.Dropout2d(0.25)

	def get_model_path(self):
		return "./noah_net.pth"

	def forward(self, x):
		'''
		Takes an image as an argument and trains that image
		'''
		x = x.view(-1, 128*128*3)
		x = F.relu(self.fc1(x))
		x = self.d25(x)
		x = F.relu(self.fc2(x))
		x = self.d10(x)
		x = F.relu(self.fc2(x))
		x = self.fc4(x)
		return x

"""
	def test(self, images):
	'''
	Takes a list of images as an argument
	Returns a list of labels predicted by the classifier
	'''
		# Load model
		self.load_state_dict(torch.load(self.path))

		# Compare with images
		outputs = self(images)

		_, predicted = torch.max(outputs, 1)

		# Print first 4 images to test
		for j in range(4):
			print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]))

		labels = []

		for i in range(len(predicted)):
			labels.append(classes[predicted[i]])

		return labels
"""