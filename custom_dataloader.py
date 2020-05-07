import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
import PIL
from PIL import Image

def pil_loader(path):
	with open(path, "rb") as f:
		img = Image.open(f)
		return img.convert('RGB')
class StringFolder(VisionDataset):
	def __init__(self, root, image_dict, categories_dict,transform=None,rescaling_size = (800,600), is_valid_file=None):
		# Much cheaper than scanning a directory (old HDD) but assumes that all files are valid...
		# Assumptions:
		#	Categories_dict is in the format {"CAT_UUID": {"id":CAT_UUID, "name":"NAME", "count":NUM},...} 
		#	image_dict is of the format {"images":[{"id":"UUID", "file_name":"xxx.jpg", "width":WIDTH, "height":HEIGHT},...], "annotations":[{"image_id":"UUID", "category_id":"CAT_ID"},...]}
		#
		#	Note if using the correct json files, image_dict === json.load(iwildcam2020_train_annotations.json)
		#         Categories_dict === json.load(Categories.json)
		# Where the json.load function is the standard load function.
		# root is where the files actually live.
		# Rescale all images to X before passing them on...
		
		super(StringFolder, self).__init__("",transform=transform, target_transform=None)
		self.transform = transform
		classes, class_to_idx, idx_to_class, idx_to_name = self._find_classes(categories_dict)
		self.resize_size = rescaling_size
		samples = self._make_dataset(root, image_dict, class_to_idx, is_valid_file)
		self.loader = pil_loader
		self.classes = classes
		self.class_to_idx = class_to_idx
		self.idx_to_class = idx_to_class
		self.idx_to_name = idx_to_name
		self.samples = samples
		self.lensamples = len(samples)
		self.ids = [s[2] for s in samples]
		self.targets = [s[1] for s in samples]
		
	def _make_dataset(self, img_root, img_dict, CTI, is_valid=None):
		instance = []
		
		# Simply runs the img_dict, add them to the instances array, and calls it a day.
		# Note that the values added are: the (full_path, class_index, id)
		# NOTE USES THE CHEAT THAT THE ORDER OF ANNOTATIONS IS THE SAME AS THE IMAGES!
		# So img_dict["annotations"][i]["image_id"] == data["images"][i]["id"] 
		# for all i.
		for img in range(len(img_dict["annotations"])):
			pth = os.path.join(img_root, str(img_dict["annotations"][img]["category_id"]), img_dict["images"][img]["file_name"])
			if is_valid == None or is_valid(pth):
				item = pth, CTI[str(img_dict["annotations"][img]["category_id"])], img_dict["images"][img]["id"]
				instance.append(item)
			# Yes that will result in a valid path, don't worry about the fact that / and \ are used indiscriminately.
		return instance
			
	def retrieve_name(self, idx):
		if idx in self.idx_to_name:
			return self.idx_to_name[idx]
		raise IndexError("IDX " + str(idx) + "NOT FOUND ERROR!")
	def _find_classes(self, dict):
		classes = list(dict.keys())
		classes.sort()
		class_to_idx = {c: i for i,c in enumerate(classes)}
		idx_to_class = {i: c for i,c in enumerate(classes)}
		idx_to_names = {i: dict[c]["name"] for i,c in enumerate(classes)}
		# Yeah...thats it.
		return classes, class_to_idx, idx_to_class, idx_to_names
		
	def __getitem__(self, index):
		path, target, id = self.samples[index]
		sample = self.loader(path)
		sample = sample.resize(self.resize_size)
		if self.transform is not None:
			sample = self.transform(sample)
		return sample,target,id
	def __len__(self):
		return self.lensamples