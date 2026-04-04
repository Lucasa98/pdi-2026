import numpy as np

class Helpers:
    def LUTLineal(a: float, c: float, rango: np.ndarray[int]|None = None)-> np.ndarray[int, float]:
        """LUT para la transformacion lineal

        Args:
            a (float): ganancia
            c (float): offset
            rango (np.ndarray[int] | None, optional): Si es None, toma [0,255]. Defaults to None.

        Returns:
            np.ndarray[int, float]: LUT s = a*r + c
        """
        grises = np.linspace(0, 255, endpoint=False) if rango == None else np.linspace(rango[0], rango[1], endpoint=False)
        return np.multiply(a,grises)+c
