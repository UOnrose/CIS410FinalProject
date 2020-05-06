import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os

from noah_classifier import Noah_Classifier
from chase_classifier import Chase_Classifier

import PIL
from PIL import Image
def sliceImage(image, bbox,targSize=(128,128)):
	# Image is assumed 
	# BBox assumed to be Left, Top, Right, Bottom
	# But fractional to the image size...
	# Returns a tensor
	(w,h) = Img.size
	return transforms.ToTensor()(transforms.functional.resized_crop(image, bbox[1]*h,bbox[0]*w, h*(bbox[3]-bbox[1]), w*(bbox[2] - bbox[0]), targSize))
	
def main():
	parser = argparse.ArgumentParser(description='Loads the classifiers and either trains them or tests them on an image.')

	# We are assuming all images will be passed/reformed to 3*128*128, the finder will be tasked with the resizing.

	parser.add_argument('-c','--chase', action='store_const', const='C', default='N', help='Use Chase\'s CNN model as a classifier. Default Noah\'s Linear Network') 
	parser.add_argument('-g', '--ground', action='store_const', const='G', default='O', help='Use the ground truth calculation as a RoI finder. Default Use our finder')
	parser.add_argument('-t','--train', action='store_const', const='TRAIN', default='TEST',help='Train the CLASSIFIERS   Default: run classifiers in testing mode')
	parser.add_argument('-r', '--roit',  action='store_const', const='ROIT', default='no', help='Train the region of interest model, takes over entire program. Default: run it in testing mode')
	parser.add_argument('-e', '--epoch', nargs='?', const=1, default=2, help='Define how many times the trainer loops over the set. Default is twice')

	parser.add_argument('-s', '--save', help='Set the save path for models being trained. Default: \'./\'') # 
	parser.add_argument('-l', '--load', help='Set the load path for loading pre-trained models. Default: \'./\'')
	parser.add_argument('image_path', help='The folder to load images from, required.')

	args = parser.parse_args()

	# Get classifier from args
	if args.chase is 'N':
		net = Noah_Classifier()
	else:
		net = Chase_Classifier()

	# Get save location for models
	if args.save is None:
		save = './'
	else:
		save = args.save

	# Get load location for models
	if args.load is None:
		load = './'
	else:
		load = args.load	

	# Fetch dataset 
	# TODO use kaggle dataset rather than CIFAR-10
	classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		
	
	data_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
	
	data_loader = torch.utils.data.DataLoader(data_set, batch_size=4,
                                          shuffle=True, num_workers=2)


	### No RoI Section ###
	# Training no RoI
	if args.train is 'TRAIN':
		# Load model
		os.chdir(load)
		try: 
			net.load_state_dict(torch.load(net.get_model_path()))
		except FileNotFoundError:
			pass
		print("Training classifiers")

		# Define loss and optimizer
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

		for epoch in range(int(args.epoch)):
			running_loss = 0.0
			
			for i, data in enumerate(data_loader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				output = net(inputs)
				loss = criterion(output, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				if i % 2000 == 1999:    # print every 2000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0

		print('Finished Training')
		# Save Model
		os.chdir(save)
		torch.save(net.state_dict(), net.get_model_path())
		print("Model saved")


	# TODO Testing no RoI
	if args.train is 'TEST':
		print("Testing classifiers")

		# Load model
		os.chdir(load)
		net.load_state_dict(torch.load(net.get_model_path()))

		dataiter = iter(data_loader)
		images, labels = dataiter.next()

		# Compare with images
		outputs = net(images)

		_, predicted = torch.max(outputs, 1)

		# Print first 4 images to test
		for j in range(4):
			print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]))

		labels = []

		for i in range(len(predicted)):
			labels.append(classes[predicted[i]])




	### RoI Section ###
	# TODO Training w/RoI


	# TODO Testing w/RoI






if __name__ == '__main__':
	main()
