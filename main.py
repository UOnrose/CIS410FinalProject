import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import json
from noah_classifier import Noah_Classifier
from chase_classifier import Chase_Classifier

from custom_dataloader import StringFolder 
import PIL
from PIL import Image
from scipy import interpolate
import multiprocessing
from itertools import product, starmap
from ground_truth_finder import GTFinder
import matplotlib.pyplot as plt

DEBUGGING_MODE_ON = False

def debug_logging(*args):
    if DEBUGGING_MODE_ON:
        print('[DEBUG: ]',*args)


## This is from https://gist.github.com/J535D165/a2ac7f27ad6ddd6ee85e7d30e9f4080e

# The problem is that we need to crop and resize images, if we do that we get a decompression issue (9billion pixels is too much)
# So instead I am using numpy :)
# This will resize it based on the above implementation. NOTE that the above does not support things like fractional cropping, and has some other issues regarding lengths.
# Also edited to support color channels :D
def crop_and_resample(arr, color_chan, x_crop, y_crop, new_size):
    d,h,w = arr.shape
    # Here is the idea, take the min/max of the bounds +- 50 and the actual bounds of the array. This will allow for us NOT to interpolate over the whole image but still not get weird 1px wide images.
    # Sort for good measure.
    x_crop.sort()
    y_crop.sort()
    x_left = int(max(0,x_crop[0]-50))
    x_right = int(min(w-1, x_crop[1]+50))
    y_top = int(max(0,y_crop[0]-50))
    y_bottom = int(min(h-1, y_crop[1]+50))
    x_dist = x_right - x_left + 1
    y_dist = y_bottom - y_top + 1
    debug_logging("Crop and Resample: ","[",x_left, x_right, y_top,y_bottom,"]", w,h, x_crop, y_crop)
    npx = np.linspace(float(x_crop[0] - x_left),x_dist-1 - (float(x_right - x_crop[1])),new_size[0])
    npy = np.linspace(float(y_crop[0] - y_top),y_dist-1 - float(y_bottom - y_crop[1]),new_size[1])
    out = np.zeros((color_chan, new_size[1],new_size[0]))
    for i in range(color_chan):
        f = interpolate.RectBivariateSpline(np.arange(y_dist), 
                                 np.arange(x_dist), 
                                 arr[i,y_top:y_bottom+1,x_left:x_right+1],kx=1,ky=1)
        out[i,:,:] = f(npy,npx)
    return out

def sliceImage(image, bbox,targSize=(128,128)):
    # Image is assumed 
    # BBox assumed to be Left, Top, Right, Bottom
    # ASSUMES THAT THOSE ARE PIXEL POSITIONS! (So [55, 67.8, 304,506.3])
    # Returns a tensor
    # print(image.size())
    d,w,h = image.size() # :|
    K = crop_and_resample(image.numpy(),d, [bbox[0],bbox[2]], [bbox[1],bbox[3]], targSize)
    ALPH = transforms.ToTensor()(np.transpose(K,(2,1,0)))
    return ALPH


def main():
    parser = argparse.ArgumentParser(description='Loads the classifiers and either trains them or tests them on an image.')

    # We are assuming all images will be passed/reformed to 3*128*128, the finder will be tasked with the resizing.

    parser.add_argument('-c','--chase', action='store_const', const='C', default='N', help='Use Chase\'s CNN model as a classifier. Default Noah\'s Linear Network') 
    parser.add_argument('-g', '--ground', action='store_const', const='G', default='O', help='Use the ground truth calculation as a RoI finder. Default Use our finder')
    parser.add_argument('-t','--train', action='store_const', const='TRAIN', default='TEST',help='Train the CLASSIFIERS   Default: run classifiers in testing mode')
    parser.add_argument('-r', '--roit',  action='store_const', const='ROIT', default='no', help='Train the region of interest model, takes over entire program. Default: run it in testing mode')
    parser.add_argument('-e', '--epoch', nargs='?', const=1, default=2, help='Define how many times the trainer loops over the set. Default is twice.')
    parser.add_argument('-n', '--image_num', nargs='?', const=1, default=20000, help='Define how many pictures the trainer will use in an iteration. Default is 20,000.')
    parser.add_argument('-d','--debug', action='store_const', const='DEBUG', default='', help='Enables debug logging.')
    # Add in parser argument for annotations directory...
    parser.add_argument('-s', '--save', help='Set the save path for models being trained. Default: \'./\'') # 
    parser.add_argument('-l', '--load', help='Set the load path for loading pre-trained models. Default: \'./\'')
    parser.add_argument('image_path', help='The folder to load images from, required.')
    parser.add_argument('--resume',action='store_const', const='Y',default='N', help='The models being loaded have been trained partially on this dataset and thus they should be modified before loading weights. Omitting this will assume the models were trained on another dataset and thus the last layer\'s weights will be dropped.')

    args = parser.parse_args()
    if args.debug is 'DEBUG':
        DEBUGGING_MODE_ON = True
    # Get classifier from args
    

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
    
    if args.chase is 'N':
        net = Noah_Classifier()
    else:
        net = Chase_Classifier()
    if args.resume is 'Y':
        if args.chase is not 'N':
            net.fc = nn.Linear(net.last_cnn_size, len(classes))
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    
    
    # TODO: Replace with argument for json locations
    # Could also be smoother/behind an if statement if not needed.
    with open("./images/iwildcam2020_train_annotations.json") as f:
        annot = json.load(f)
    with open("./images/Categories.json") as f:
        cate = json.load(f)
        
    data_set = StringFolder(root=args.image_path, image_dict = annot, categories_dict = cate, transform=transform)
    
    #, transform=transform) # This transform will break my image cropper (as it assumed it ISN'T a tensor
    
    regionProposer = GTFinder(dir = './images/')
    
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=4,
                                          shuffle=True, num_workers=4)

    pool = multiprocessing.Pool()
    ### No RoI Section ###
    # Training no RoI
    if args.train is 'TRAIN':
        # Load model
        os.chdir(load)
        try: 
            net.load_state_dict(torch.load(net.get_model_path()))
            if args.chase is not 'N' and args.resume is 'N':
                # We are going to be fine-tuning this model.
                net.fc = nn.Linear(net.last_cnn_size, len(classes))
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
                inputs, labels, ids = data
                # TODO ADD IN THE TOOL THAT GETS THE BBOX OF ALL THE IMAGES!
                # NOTE THAT THIS CAN'T BE PARALLIZED EASILY! WE WILL NEED A FORLOOP
                
                bboxes = regionProposer.search(inputs, ids)
                #image, bbox,
                cboxes = [] # Idea is that this is a set of tuples, each item in tuple as an input to be mapped over sliceImage
                clabels = []
                for kk in range(len(inputs)):
                    cboxes.extend(product([inputs[kk]], [j[0] for j in bboxes[kk]]))
                    clabels.extend([labels[kk] for k in range(len(bboxes[kk]))])
                if len(clabels) == 0:
                    debug_logging("The amount of labels in the list are zero. This means we got unlucky with our draw and to continue to the next set.")
                    continue 
                clabels = torch.from_numpy(np.asarray(clabels))
                # If multiprocessing wasn't haven't a heart attack, I would do that to be faster...only better if more multiprocessing
                results_temp = list(starmap(sliceImage, cboxes))  
                results = torch.zeros([len(results_temp), 3,128,128],dtype=torch.float)
                for i in range(len(results_temp)):
                    results[i,:,:,:] = results_temp[i]
                #ii = sliceImage(inputs[0], [0, 0, .5,.5])
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = net(results)
                loss = criterion(output, clabels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 19))
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
        images, labels,ids = dataiter.next()
        # TODO ADD IN THE TOOL THAT GETS THE BBOX OF ALL THE IMAGES!
        # NOTE THAT THIS CAN'T BE PARALLIZED EASILY! WE WILL NEED A FORLOOP
        ii = sliceImage(inputs[0], [0, 0, .5,.5])
        print(inputs)
        print(ii)
        # XXX
        return inputs
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
    _temp = main()
