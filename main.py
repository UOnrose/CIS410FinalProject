import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import json
from noah_classifier import Noah_Classifier
from chase_classifier import Chase_Classifier

from custom_dataloader import StringFolder 
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
	parser.add_argument('-e', '--epoch', nargs='?', const=1, default=2, help='Define how many times the trainer loops over the set. Default is twice.')
	parser.add_argument('-i', '--image_num', nargs='?', const=1, default=20000, help='Define how many pictures the trainer will use in an iteration. Default is 20,000.')
	# Add in parser argument for annotations directory...
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
	classes = (0, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 20, 21, 24, 25, 26, 31, 32, 37, 38, 40, 41, 44, 50, 51, 62, 67, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 83, 86, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 127, 129, 130, 133, 134, 137, 139, 141, 142, 144, 145, 147, 150, 152, 153, 154, 156, 159, 161, 162, 163, 166, 167, 170, 175, 177, 179, 188, 198, 210, 221, 227, 229, 230, 233, 234, 235, 240, 242, 243, 245, 250, 251, 252, 253, 256, 257, 258, 259, 262, 265, 267, 268, 273, 279, 286, 290, 291, 292, 294, 296, 299, 300, 301, 302, 306, 307, 309, 310, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 352, 353, 354, 355, 356, 357, 370, 371, 372, 374, 375, 376, 377, 378, 379, 380, 382, 384, 385, 389, 390, 391, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 422, 454, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 599, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675)
	
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	
	
	
	# TODO: Replace with argument for json locations
	# Could also be smoother/behind an if statement if not needed.
	with open("./images/iwildcam2020_train_annotations.json") as f:
		annot = json.load(f)
	with open("./images/Categories.json") as f:
		cate = json.load(f)
		
		
	print("NEAR")
	data_set = StringFolder(root=args.image_path, image_dict = annot, categories_dict = cate)
	print("HERE")
	print("LEN: " + str(len(data_set)))
	print("'RANDOM' SAMPLING:")
	RS = [1, 14592, 312, 4152, 2222, 665]
	for i in RS:
		print(data_set[i])
	return
	#, transform=transform) # This transform will break my image cropper (as it assumed it ISN'T a tensor
	
	
	
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
			image_count = 0

			for i, data in enumerate(data_loader, 0):
				# max number of images trained
				if image_count > args.image_num:
					break

				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				inputs = sliceImage(inputs, [0, 0, inputs.width, inputs.height])

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

				# Keep track of amount of images trained
				image_count += 1

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
