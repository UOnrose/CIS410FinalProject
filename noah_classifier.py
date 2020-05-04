from chase_classifier import Chase_Classifier

class Noah_Classifier(Chase_Classifier):
	def __init__(self):
	'''
	Initializer for Noah_Classifier
	creates linear layers and dropouts
	'''
		super(Noah_Classifier, self).__init__()

		# Save location for model
		self.path = './noah_net.pth'

		# Save blank model
		torch.save(self.state_dict(), self.path)

		# Linear Layers
		self.fc1 = nn.Linear(128*3*3, 300)
		self.fc2 = nn.Linear(300, 300)
		self.fc3 = nn.Linear(300, 300)
		self.fc4 = nn.Linear(300, 200)

		# Dropouts
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


	def train(self, images, labels):
	'''
	Takes a list of images and their labels as arguments
	Trains with those images
	'''
		# Load model
		self.load_state_dict(torch.load(self.path))

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

		# Train with given images
		for epoch in range(2): # Loop over dataset twice
			running_loss = 0.0
			for i in range(len(images)):
				img = images[i]
				label = labels[i]

				# zero the parameter gradients
		        optimizer.zero_grad()

		        # forward + backward + optimize
		        output = self(img)
		        loss = criterion(output, label)
		        loss.backward()
		        optimizer.step()

		        # print statistics
		        running_loss += loss.item()
		        if i % 2000 == 1999:    # print every 2000 mini-batches
		            print('[%d, %5d] loss: %.3f' %
		                  (epoch + 1, i + 1, running_loss / 2000))
		            running_loss = 0.0

        # Save new model
        torch.save(self.state_dict(), self.path)

		print('Finished Training')	