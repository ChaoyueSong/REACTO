import numpy as np
import cv2


UNKNOWN_FLOW_THRESH = 1e7


def warp_flow(img, flow, normed=False):
    """
    inverse warping. img is the 2nd frame. flow is 1->2.
    output: warped 2nd frame in the coordinate of 1st frame.
    """
    h, w = flow.shape[:2]
    flow = flow.copy().astype(np.float32)
    if normed:
        flow[:, :, 0] = flow[:, :, 0] * w / 2.0
        flow[:, :, 1] = flow[:, :, 1] * h / 2.0
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def point_vec(img, flow, skip=40):
    assert flow.ndim == 3  # h, w, 3
    assert img.ndim == 3  # h, w, 3
    assert flow.shape[2] == 2 or flow.shape[2] == 3
    if flow.shape[-1] == 2:
        invalid = np.linalg.norm(flow, 2, -1) == 0
        flow = np.concatenate([flow, invalid.astype(float)[..., None]], -1)
    if img.shape[-1] == 4:
        transparency = img[..., 3]
        img = img[..., :3]

    extendfac = 1.0
    resize_factor = 1
    # resize_factor = max(1,int(max(maxsize/img.shape[0], maxsize/img.shape[1])))
    meshgrid = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    dispimg = cv2.resize(
        img[:, :, ::-1].copy(), None, fx=resize_factor, fy=resize_factor
    )
    colorflow = flow_to_image(flow).astype(int)
    for i in range(img.shape[1]):  # x
        for j in range(img.shape[0]):  # y
            if flow[j, i, 2] > 0:  # uncertainty from -inf to inf
                continue
            if j % skip != 0 or i % skip != 0:
                continue
            xend = int((meshgrid[0][j, i] + extendfac * flow[j, i, 0]) * resize_factor)
            yend = int((meshgrid[1][j, i] + extendfac * flow[j, i, 1]) * resize_factor)
            leng = np.linalg.norm(flow[j, i, :2] * extendfac)
            if leng < 1:
                continue
            dispimg = cv2.arrowedLine(
                dispimg,
                (meshgrid[0][j, i] * resize_factor, meshgrid[1][j, i] * resize_factor),
                (xend, yend),
                (
                    int(colorflow[j, i, 2]),
                    int(colorflow[j, i, 1]),
                    int(colorflow[j, i, 0]),
                ),
                1,
                tipLength=4 / leng,
                line_type=cv2.LINE_AA,
            )
    if img.shape[-1] == 4:
        # add back transparency channel
        transparency = cv2.resize(
            transparency, None, fx=resize_factor, fy=resize_factor
        )
        transparency[np.abs(dispimg).sum(-1) > 0] = 255
        dispimg = np.concatenate([dispimg, transparency[..., None]], -1)
    return dispimg


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col : col + YG, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, YG) / YG)
    )
    colorwheel[col : col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col : col + CB, 1] = 255 - np.transpose(
        np.floor(255 * np.arange(0, CB) / CB)
    )
    colorwheel[col : col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col : col + MR, 2] = 255 - np.transpose(
        np.floor(255 * np.arange(0, MR) / MR)
    )
    colorwheel[col : col + MR, 0] = 255

    return colorwheel
