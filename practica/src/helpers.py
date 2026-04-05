import numpy as np
import cv2

def LUTLineal(a: float, c: float, rango: np.ndarray[int] = [0, 256]) -> np.ndarray[int, float]:
    """LUT para la transformacion lineal

    Args:
        a (float): ganancia
        c (float): offset
        rango (np.ndarray[int], optional): Defaults to [0,256].

    Returns:
        np.ndarray[int, float]: LUT s = a*r + c
    """
    grises = np.linspace(rango[0], rango[1], rango[1]-rango[0], endpoint=False)
    return np.multiply(a,grises)+c

def breakpoints2LUT(p1: tuple[int, float], p2: tuple[int, float]) -> np.ndarray[int, float]:
    """LUT para una lineal por tramos que comienza en (0,0), pasa por p1, p2 y termina en (255,255)

    Args:
        p1 (tuple[int, float]): punto 1
        p2 (tuple[int, float]): punto 2

    Returns:
        np.ndarray[int, float]: LUT por tramos
    """
    # ganancias (pendientes)
    a0 = p1[1]/p1[0]
    a1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    a2 = (255 - p2[1]) / (255 - p2[0])

    # offsets (el primer tramo tiene offset 0)
    c1 = p1[1] - a1*p1[0]
    c2 = p2[1] - a2*p2[0]

    return np.concat([
        LUTLineal(a0, 0, np.array([0, p1[0]], dtype=int)),
        LUTLineal(a1, c1, np.array([p1[0], p2[0]], dtype=int)),
        LUTLineal(a2, c2, np.array([p2[0], 256], dtype=int)),
    ])

def LUTLog(rango: np.ndarray[int] = [0, 256]) -> np.ndarray[int, float]:
    """LUT para transformacion logaritmica

    Args:
        rango (np.ndarray[int], optional): Defaults to [0, 256].

    Returns:
        np.ndarray[int, float]: LUT s = log(1+r)
    """
    grises = np.linspace(rango[0], rango[1], rango[1]-rango[0], endpoint=False)
    return np.log(1+grises)

def LUTPotencia(gamma: float, rango: np.ndarray[int] = [0, 256]) -> np.ndarray[int, float]:
    """LUT para transformacion de potencia

    Args:
        gamma (float): potencia
        rango (np.ndarray[int], optional): Defaults to [0, 256].

    Returns:
        np.ndarray[int, float]: LUT s = r^(gamma)
    """
    grises = np.linspace(rango[0], rango[1], rango[1]-rango[0], endpoint=False)
    grises_norm = grises/255.0
    # generar lut y reescalar a [0,255]
    return np.power(grises_norm, gamma) * 255.0

def promedio(imgs: np.ndarray[cv2.typing.MatLike]) -> cv2.typing.MatLike:
    """Promedio sumando y normalizando por el numero de imagenes

    Args:
        imgs (np.ndarray[cv2.typing.MatLike]): imagenes

    Returns:
        cv2.typing.MatLike: imagen promedio
    """
    img = imgs.sump(axis=0)
    return img[:]/imgs.shape[0]

def promedioVideo(vid: cv2.VideoCapture, frames: int) -> cv2.typing.MatLike:
    max_frames = min(frames, int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
    count = 0
    img_final = None
    success, img = vid.read()
    while success and count < max_frames:
        if img_final is None:
            img_final = np.float32(img)
        else:
            img_final += img
        success, img = vid.read()
        count += 1
    img_final /= max_frames
    return np.uint8(img_final)

def diferencia(img1: cv2.typing.MatLike, img2: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """diferencia de dos imagenes reescalando para evitar desborde

    Args:
        img1 (cv2.typing.MatLike)
        img2 (cv2.typing.MatLike)

    Returns:
        cv2.typing.MatLike: diferencia img1 - img2
    """
    img = img1 - img2
    img = (img + 255)/2
    # metodo 2
    #img = (img - np.min(img))/255
    return img

def multiplicacion(img: cv2.typing.MatLike, mascara: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Para extraccion de ROI

    Args:
        img (cv2.typing.MatLike): imagen
        mascara (cv2.typing.MatLike): mascara binaria

    Returns:
        cv2.typing.MatLike: ROI
    """
    return np.multiply(img, mascara)
