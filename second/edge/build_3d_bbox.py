# -*- coding: utf-8 -*-
# Filename : build_3d_bbox
__author__ = 'Xumiao Zhang'

import sys
import os
sys.path.append("..")
sys.path.append("../..")
import numpy as np
import torchplus
import torch

data_path = '/home/xumiao/Edge/test/occlusion_scenario/0819/'  # data_path = '/home/xumiao/Edge/inference/' 0407  car
save_path = '/home/xumiao/Edge/test/occlusion_scenario/0819/'  # save_path = '/home/xumiao/Edge/bbox/' 0407  car


def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, s,  0], [-s,  c,  0], [0,  0,  1]])

def interpolation(a, b):
# 15/30 points, for short edges
	num_points = 30
	step = np.zeros(3)
	points = np.empty((num_points,3), dtype=np.float32)
	for i in range(3):
		step[i] = (b[i]-a[i]) / (num_points+1)
	for i in range(0,num_points):
	 	points[i] = np.float32([a[0] + (i+1)*step[0], a[1] + (i+1)*step[1], a[2] + (i+1)*step[2]])
	return points
def interpolation2(a, b):
# 40/70 points, for long edges
	num_points = 70
	step = np.zeros(3)
	points = np.empty((num_points,3), dtype=np.float32)
	for i in range(3):
		step[i] = (b[i]-a[i]) / (num_points+1)
	for i in range(0,num_points):
	 	points[i] = np.float32([a[0] + (i+1)*step[0], a[1] + (i+1)*step[1], a[2] + (i+1)*step[2]])
	return points

def buildBox(preds):
	boxs = np.empty((0,3), dtype=np.float32)
	
	for box_para in preds:
		if box_para[7] > 0.4: # scoce > 0.3
			h, w, l, x, y, z = box_para[0:6]
			R = rotz(box_para[6]) # rotation matrix
			
			x_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
			y_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
			z_corners = [0,0,0,0,h,h,h,h]
			
			corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
			
			corners_3d[0,:] = corners_3d[0,:] + x;
			corners_3d[1,:] = corners_3d[1,:] + y;
			corners_3d[2,:] = corners_3d[2,:] + z;
			
			box = np.float32(np.transpose(corners_3d))
			
			for i, j in zip([0,2,4,6,0,1,2,3], [1,3,5,7,4,5,6,7]):
				points = interpolation(box[i], box[j])
				box = np.vstack((box,points))
			for i, j in zip([1,3,5,7], [2,0,6,4]):
				points = interpolation2(box[i], box[j])
				box = np.vstack((box,points))
			
			boxs = np.vstack((boxs,box))
	return boxs

if __name__ == '__main__':
	for fold in os.listdir(data_path):
		if fold.split('.')[1] == 'txt':
			print(data_path + fold)
			preds = np.loadtxt(data_path + fold)
			boxs = buildBox(preds)

			with open(save_path + fold.split('.')[0] + '_box.bin', 'w') as f:
				boxs.tofile(f)

	# preds = np.loadtxt(data_path + sys.argv[1] +'.txt')
	# boxs = buildBox(preds)

	# #print(np.shape(boxs))
	# with open(save_path + sys.argv[1] + '_box.bin', 'w') as f:
	#     boxs.tofile(f)
