import numpy as np
import math
from torchvision.transforms import functional as trf

def lip_points(pts_file):
    with open(pts_file) as f:
        pts = f.readlines()
    pts = [pts[48], pts[54]]
    pts = [pt.strip().split() for pt in pts]
    pts = [[float(pt[0]), float(pt[1])] for pt in pts]
    pts = np.array(pts)
    return pts


def lip_region(image, pts):
    left, right = pts[0], pts[1]
    centre = np.average(pts, axis=0)
    vect = right - left
    angle = math.degrees(math.atan(vect[1] / vect[0]))
    distance = np.linalg.norm(vect)
    image = trf.rotate(image, angle, center=centre.tolist())
    box = [centre[0] - distance/2 * 1.2, centre[1] - distance/2 * 0.8,   # width, height
           centre[0] + distance / 2 * 1.2, centre[1] + distance / 2 * 0.8]  # width, height
    image = image.crop(box)
    return image


import os
from PIL import Image
data = []
directory = 'staff_makeup/train'
classes = os.listdir(directory)
classes.sort()
for i, c in enumerate(classes):
    images = os.listdir(os.path.join(directory, c))
    images = [os.path.join(directory, c, img) for img in images if '.jpg' in img]
    data += [{'image': img,
            'lip_region_img': lip_points(img.replace('.jpg', '.pts')),
            'label': i} for img in images]

    for img in images:
        pts = lip_points(img.replace('.jpg', '.pts'))
        image = Image.open(img).convert('RGB')
        image = lip_region(image, pts)
        # image.show()
        image.save(img.replace('.jpg', '_lip.jpg'))

    

