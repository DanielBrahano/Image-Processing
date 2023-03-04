# Student 1 ID: 208869222
# Student 2 ID: 205465107

import numpy as np
import matplotlib.pyplot as plt
import cv2


def histImage(im):
    h = np.zeros(256, np.float)

    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            h[int(im[i][j])] += 1

    return h


def nhistImage(im):
    nh = np.zeros(256, np.float)
    pixel_num = im.shape[0] * im.shape[1]

    nh = histImage(im)

    for i in range(0, len(nh)):
        nh[i] = float(nh[i]/pixel_num)

    return nh


def ahistImage(im):
    ah = np.zeros(256, np.float)
    h = histImage(im)

    ah[0] = h[0]
    for i in range(1, len(ah)):
        ah[i] = ah[i-1]+h[i]
    return ah


def calcHistStat(h):
    # m = np.dot(h, np.arange(0, 256, 1))/np.sum(h)
    # e = np.dot(h, np.square(np.arange(0, 256, 1)))/np.sum(h) - (m * m)

    m = np.dot(h, np.arange(0, 256, 1))/np.sum(h)
    e = np.dot(h, np.square(np.arange(0, 256, 1))) / np.sum(h) - (m * m)



    return m, e


def mapImage(im, tm):
  #  nim = np.zeros((im.shape[0], im.shape[1]), np.float)
    tm1 = np.zeros(len(tm))
    for i in range(0, 256):
        if tm[i] > 255:
            tm1[i] = 255
        elif tm[i] < 0:
            tm1[i] = 0
        else:
            tm1[i] = tm[i]

  #  for i in range(0, 255):
  #      nim[im == i] = tm1[i]
  #  return nim
    return tm1[im]


def histEqualization(im):
    ah = ahistImage(im)
    goal_histogram = np.empty(ah.shape)
    goal_histogram = np.full(256, ah[255]/ah.shape)
    acc_goal_histogram = np.cumsum(goal_histogram)
    tm = np.zeros(256)
    k = 0
    for i in range(0, 256):
        while k < 256:
            if ah[i] <= acc_goal_histogram[k]:
                tm[i] = k
                break
            else:
                k = k + 1

    return tm
