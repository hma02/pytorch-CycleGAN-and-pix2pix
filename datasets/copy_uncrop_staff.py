from glob import glob

# trainA = glob('staff_makeup/trainA/*.jpg')

# trainA = [t.split('/')[-1] for t in trainA]

# print(trainA)

mnt_path_lipstick = '/mnt/Data/Data/lipstick/single_lipstick_staff/lipstick/'
mnt_path_no_lipstick = '/mnt/Data/Data/lipstick/single_lipstick_staff/no_lipstick/'

from shutil import copyfile
import os.path


def copy_file(target, folder):

    fname = mnt_path_lipstick+target

    if os.path.isfile(fname):
        pass
    else:
        fname = mnt_path_no_lipstick+target
        assert os.path.isfile(fname)

    if not os.path.exists('./staff_makeup_nocrop/'+folder):
        os.makedirs('./staff_makeup_nocrop/'+folder)
    target_fname = './staff_makeup_nocrop/'+folder + '/'+target
    copyfile(fname, target_fname)

    print(target_fname)

    return target_fname


def rotate(fname):

    from PIL import Image
    from PIL import ExifTags

    image = Image.open(fname)

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = dict(image._getexif().items())

    if exif[orientation] == 3:
        image = image.rotate(180, expand=True)
    elif exif[orientation] == 6:
        image = image.rotate(270, expand=True)
    elif exif[orientation] == 8:
        image = image.rotate(90, expand=True)

    image.save(fname)


for folder in ['trainA', 'trainB', 'testA', 'testB', 'valA', 'valB']:
    file_full_paths = glob('staff_makeup/'+folder+'/*.jpg')
    file_names = [t.split('/')[-1] for t in file_full_paths]
    for f in file_names:
        t_f = copy_file(f, folder)
        rotate(t_f)
