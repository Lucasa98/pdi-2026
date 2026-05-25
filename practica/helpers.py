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

def kernelCruzPromediado(size: int):
    """Crear kernel de promediado en cruz

    Args:
        size (int): tamanio
    """
    # asegurar size impar
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.float32)
    centro = size // 2

    kernel[centro, :] = 1
    kernel[:, centro] = 1

    # Normalzar pesos
    kernel /= kernel.sum()

    return kernel

def kernelImg(kernel):
    h, w = kernel.shape

    # fondo negro
    kernel_img = np.zeros((127, 127, 3), dtype=np.uint8)

    # centro
    ini_y = 127 // 2 - h // 2
    ini_x = 127 // 2 - w // 2

    # dibujar kernel
    roi = kernel_img[ini_y:ini_x+h, ini_x:ini_x+w]
    roi[kernel > 0] = [0, 255, 0]

    return kernel_img

def altaPotencia(img, kernelSize=(3,3), sigma=1, A=1):
    return A*img - cv2.GaussianBlur(img, kernelSize, sigmaX=sigma, sigmaY=sigma)

def apply_filter(img, mascara):
    # transformada
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # centrar
    dft_shift = np.fft.fftshift(dft)
    # aplicar filtro
    filtered_dft = dft_shift * mascara[:, :, np.newaxis]
    # desplazar a ubicacion original
    f_ishift = np.fft.ifftshift(filtered_dft)
    # dft inversa
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    # normalizar
    img_back = np.clip(img_back / np.max(img_back), 0, 1)
    return img_back

def apply_pb_ideal(img, freq):
    # centro de la imagen
    c_row, c_col = img.shape[0]//2, img.shape[1]//2
    # circulo centrado y de radio igual a la frecuencia de corte
    mascara = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    cv2.circle(mascara, (c_col, c_row), freq, 1, thickness=-1)

    return apply_filter(img, mascara)

def apply_pb_butterworth(img, freq, orden):
    c_row, c_col = img.shape[0] // 2, img.shape[1] // 2
    mascara = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for u in range(img.shape[0]):
        for v in range(img.shape[1]):
            # distancia al centro
            distance = np.sqrt((u - c_row) ** 2 + (v - c_col) ** 2)
            mascara[u, v] = 1 / (1 + (distance / freq) ** (2 * orden))

    return apply_filter(img, mascara)

def apply_pb_gaussiano(img, tam, sigma):
    c_col = tam[1]//2
    c_row = tam[0]//2
    # malla
    y, x = np.ogrid[-c_row : c_row+1, -c_col : c_col+1]
    # filtro
    mascara = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # dft del filtro a escala de la imagen
    filter_dft = np.fft.fft2(mascara, img.shape)
    filter_dft_shift = np.fft.fftshift(filter_dft)

    # dft de la imagen
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    # aplicar filtro
    filtered_dft_shift = dft_shift * filter_dft_shift

    # invertir
    filtered_dft = np.fft.ifftshift(filtered_dft_shift)
    filtered = np.fft.ifft2(filtered_dft)

    return np.abs(filtered)
