import cv2
import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Trimap')
    parser.add_argument('--mskDir', type=str, required=True, help="masks directory")
    parser.add_argument('--saveDir', type=str, required=True, help="where trimap result save to")
    parser.add_argument('--list', type=str, required=True, help="list of images id")
    parser.add_argument('--size', type=int, required=True, help="kernel size")
    args = parser.parse_args()
    print(args)
    return args

def erode_dilate(msk, struc="ELLIPSE", size=(10, 10)):
    if struc == "RECT":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif struc == "CORSS":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    #msk = msk.astype(np.float32)
    msk = msk / 255
    #msk = msk.astype(np.uint8)

    # val in 0 or 255

    dilated = cv2.dilate(msk, kernel, iterations=1) * 255
    eroded = cv2.erode(msk, kernel, iterations=1) * 255

    cnt1 = len(np.where(msk >= 0)[0])
    cnt2 = len(np.where(msk == 0)[0])
    cnt3 = len(np.where(msk == 1)[0])
    #print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert(cnt1 == cnt2 + cnt3)

    cnt1 = len(np.where(dilated >= 0)[0])
    cnt2 = len(np.where(dilated == 0)[0])
    cnt3 = len(np.where(dilated == 255)[0])
    #print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert(cnt1 == cnt2 + cnt3)

    cnt1 = len(np.where(eroded >= 0)[0])
    cnt2 = len(np.where(eroded == 0)[0])
    cnt3 = len(np.where(eroded == 255)[0])
    #print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert(cnt1 == cnt2 + cnt3)

    res = dilated.copy()
    #res[((dilated == 255) & (msk == 0))] = 128
    res[((dilated == 255) & (eroded == 0))] = 128

    return res

def main():
    args = get_args()
    #f = open(args.list)
    #names = f.readlines()
    names = os.listdir(args.mskDir)
    print("Images Count: {}".format(len(names)))
    for name in names:
        msk_name = args.mskDir + "/" + name
        trimap_name = args.saveDir + "/" + name
        msk = cv2.imread(msk_name, -1)
        if msk is None:
            print('Load error: ', msk_name)
            continue
        msk = msk[:, :, -1]
        msk[msk>0] = msk.max()
        #print('msk.shape: ', msk.shape)
        trimap = erode_dilate(msk, size=(args.size,args.size))

        print("Write to {}".format(trimap_name))
        cv2.imwrite(trimap_name, trimap)

if __name__ == "__main__":
    main()


