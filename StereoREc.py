import stereo
import scipy.misc
from PIL import Image
from pylab import *
from scipy.ndimage import *
import numpy as np


def plane_sweep_ncc(im_l, im_r, start, steps, wid):
    # Используйте нормализованную кросс-корреляцию для вычисления несоответствия изображения "" "
    m, n = im_l.shape
    # Сохраняем массив различных значений суммы
    mean_l = zeros((m, n))
    mean_r = zeros((m, n))
    s = zeros((m, n))
    s_l = zeros((m, n))
    s_r = zeros((m, n))
    # Сохраняем массив плоскости глубины
    dmaps = zeros((m, n, steps))
    # Вычислить среднее значение блока изображения
    filters.uniform_filter(im_l, wid, mean_l)
    filters.uniform_filter(im_r, wid, mean_r)
    # Нормализованное изображение
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    # Попробуйте разные параллаксы
    for displ in range(steps):
        # Сдвинуть левое изображение вправо, вычислить сумму
        filters.uniform_filter(roll(norm_l, -displ - start) * norm_r, wid, s)  # И нормализация
        filters.uniform_filter(roll(norm_l, -displ - start) * roll(norm_l, -displ - start), wid, s_l)
        filters.uniform_filter(norm_r * norm_r, wid, s_r)  # И денормализация
        # Сохранить счет ncc
        dmaps[:, :, displ] = s / sqrt(s_l * s_r)
    # Выберите лучшую глубину для каждого пикселя
    return argmax(dmaps, axis=2)


def plane_sweep_gauss(im_l, im_r, start, steps, wid):
    # Используйте нормализованную кросс-корреляцию с периферией, взвешенной по Гауссу, для вычисления параллакса изображения "" "
    im_l = array(Image.open(im_l).convert('L'), 'f')
    im_r = array(Image.open(im_r).convert('L'), 'f')
    m, n = im_l.shape
    # Сохраняем массив разных дополнений
    mean_l = zeros((m, n))
    mean_r = zeros((m, n))
    s = zeros((m, n))
    s_l = zeros((m, n))
    s_r = zeros((m, n))
    # Сохраняем массив плоскости глубины
    dmaps = zeros((m, n, steps))
    # Рассчитать среднее
    filters.gaussian_filter(im_l, wid, 0, mean_l)
    filters.gaussian_filter(im_r, wid, 0, mean_r)
    # Нормализованное изображение
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    # Попробуйте разные параллаксы
    for displ in range(steps):
        # Сдвинуть левое изображение вправо, вычислить сумму
        filters.gaussian_filter(roll(norm_l, -displ - start) * norm_r, wid, 0, s)  # И нормализация

        filters.gaussian_filter(roll(norm_l, -displ - start) * roll(norm_l, -displ - start), wid, 0, s_l)
        filters.gaussian_filter(norm_r * norm_r, wid, 0, s_r)  # И денормализация
    # Сохранить счет ncc
    dmaps[:, :, displ] = s / sqrt(s_l * s_r)
    # Выберите лучшую глубину для каждого пикселя
    plt.imshow(argmax(dmaps, axis=2))
    plt.show()
    return argmax(dmaps, axis=2)



    #im_l = array(Image.open('d:/picture/008/im2.ppm').convert('L'), 'f')
    #im_r = array(Image.open('d:/picture/008/im6.ppm').convert('L'), 'f')
    # Начальное смещение и установите размер шага
    #steps = 50
    #start = 4

    # ширина ncc
    #wid = 13

    #res = plane_sweep_ncc(im_l, im_r, start, steps, wid)

    #imsave('d:/picture/008/depyh6.jpg', res)