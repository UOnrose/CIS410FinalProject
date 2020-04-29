import argparse

parser = argparse.ArgumentParser(description='Loads the classifiers and either trains them or tests them on an image.')

# We are assuming all images will be passed/reformed to 3*128*128, the finder will be tasked with the resizing.

parser.add_argument('-c','--chase', action='store_const', const='N', default='C', help='Use Chase\'s CNN model as a classifier. Default Noah\'s Linear Network') 
parser.add_argument('-g', '--ground', action='store_const', const='G', default='O', help='Use the ground truth calculation as a RoI finder. Default Use our finder')
parser.add_argument('-t','--train', action='store_const', const='TRAIN', default='TEST',help='Train the CLASSIFIERS   Default: run classifiers in testing mode')
parser.add_argument('-r', '--roit',  action='store_const', const='ROIT', default='no', help='Train the region of interest model, takes over entire program. Default: run it in testing mode')

parser.add_argument('-s', '--save', help='Set the save path for models being trained. Default: \'./\'') # 
parser.add_argument('-l', '--load', help='Set the load path for loading pre-trained models. Default: \'./\'')
parser.add_argument('image_path', help='The folder to load images from, required.')

parser.parse_args()


