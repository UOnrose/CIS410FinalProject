from chase_classifier import Chase_Classifier

class Noah_Classifier(Chase_Classifier):
	def __init__(self):
	'''
	Initializer for Noah_Classifier
	creates linear layers and dropouts
	'''
		super(Noah_Classifier, self).__init__()
		self.fc1 = nn.Linear(128*3*3, 300)
		self.fc2 = nn.Linear(300, 300)
		self.fc3 = nn.Linear(300, 300)
		self.fc4 = nn.Linear(300, 200)
		self.d10 = nn.Dropout2d(0.1)
		self.d25 = nn.Dropout2d(0.25)

	def forward(self, x):
	'''
	Takes an image as an argument and trains that image
	'''
		x = F.relu(self.fc1(x))
		x = self.d25(x)
		x = F.relu(self.fc2(x))
		x = self.d10(x)
		x = F.rel(self.fc3(x))
		x = self.fc4(x)


	def test(self, images):
	'''
	Takes a list of images as an argument
	Returns a list of labels predicted by the classifier
	'''
		pass

	def train(self, images, labels):
	'''
	Takes a list of images and their labels as arguments
	Trains with those images
	'''
		pass
	