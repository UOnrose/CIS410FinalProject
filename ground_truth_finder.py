import json
import os
class GTFinder():
	def __init__(self,dir):
		# The idea is that we will load up the json file, wait for us to find the image, and call it a day...
		with open(os.path.join(dir,"iwildcam2020_megadetector_results.json")) as f:
			jsonfile = json.load(f)
		self.img_id_lookup = {}
		for val in jsonfile["images"]:
			self.img_id_lookup[val["id"]] = val["detections"]
	def search(self, images, img_ids):
		# Image id is EXTENSIVELY used here
		# This will return a list of bounding boxes (along with confidence scores) of detections...
		# [] is an acceptable answer.
		# Detections will be in [[(bbox1, conf1),(bbox2, conf2),...],...], with each outer matrix being for each image in the batch.
		# Where each bbox will be left, top, right, bottom.
		# Need to get size of image....
		
		
		# Remember that the width of the image is the length of the third index
		# The height is the length of the second index. (so ID,Channel,Row,Col)
		bboxes = [[] for i in range(len(img_ids))]
		for id in range(len(img_ids)):
			img_i = img_ids[id]
			img_w = len(images[id][0][0])
			img_h = len(images[id][0])
			if img_i in self.img_id_lookup:
				dets = self.img_id_lookup[img_i]
				for img_id in dets:
					bboxes[id].append(([img_id["bbox"][0] * img_w, img_id["bbox"][1]*img_h, 
					(img_id["bbox"][2]+img_id["bbox"][0])*img_w, img_h*(img_id["bbox"][1] + img_id["bbox"][3])],
					img_id["conf"]))
		return bboxes