#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 12:00:20 2023

@author: mines46
"""

import numpy as np
import matplotlib.pyplot as plt

angle = np.linspace(0, 2*np.pi*10, 1000)

sin = np.sin(angle)
cos = np.cos(angle)
tan = np.tan(angle)

sin_pi4 = np.sin(angle+np.pi/4)
cos_pi4 = np.cos(angle+np.pi/4)
tan_pi4 = np.tan(angle+np.pi/4)

plt.close('all')

fig, axels = plt.subplots(3, 1, figsize=(8,12))


axels[0].plot(angle, cos, "r", label = "alpha")
axels[0].plot(angle, cos_pi4, "k--", label = "alpha+pi/4")
axels[0].axis([0, 20, -1, 1])
axels[0].set_title("cosine function")
axels[0].legend()
axels[0].grid(True)

axels[1].plot(angle, sin, "r", label = "alpha")
axels[1].plot(angle, sin_pi4, "k--", label = "alpha+pi/4")
axels[1].axis([0, 20, -1, 1])
axels[1].set_title("sine function")
axels[1].legend()
axels[1].grid(True)

axels[2].plot(angle, tan, "r", label = "alpha")
axels[2].plot(angle, tan_pi4, "k--", label = "alpha+pi/4")
axels[2].axis([0, 20, -10, 10])
axels[2].set_title("tangent function")
axels[2].legend()
axels[2].grid(True)