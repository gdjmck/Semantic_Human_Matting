import os
import pickle
#import cv2

matting_files = []
img_files = []

current_folder = ['matting_human_half/clip_img']
while len(current_folder) > 0:
    search_folder = current_folder[0]
    current_folder.remove(search_folder)
    canadate = os.listdir(search_folder)
    for f in canadate:
        f = os.path.join(search_folder, f)
        if os.path.isfile(f):
            img_files.append(f)
            matting_files.append(f.replace('clip_img', 'matting').replace('clip', 'matting').strip()[:-3]+'png')
        else:
            current_folder.append(f)
         
print(len(matting_files))
'''
# test
for i in range(10):
    matte = cv2.imread(matting_files[i], -1)
    img = cv2.imread(img_files[i], -1)
    print(matting_files[i], '\n', img_files[i])
    assert matte.shape[0] == img.shape[0] and matte.shape[1] == img.shape[1]
'''
'''
with open('imgList.pkl', 'wb') as f:
    pickle.dump(img_files, f)
with open('imgMatte.pkl', 'wb') as f:
    pickle.dump(matting_files, f)
'''