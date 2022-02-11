"""
All methods/classes related to image handling
"""


def crop_frac(img, fracx, fracy):
    z, y, x = img.shape
    cropx = int(x * fracx)
    cropy = int(y * fracy)
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty + cropy, startx:startx + cropx]
