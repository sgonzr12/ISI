#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:05:40 2023

@author: mines46
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas

sales_df = pandas.read_csv("company_sales_data.csv")
months_df = sales_df["month_number"]
facecream_df = sales_df["facecream"]
facewash_df = sales_df["facewash"]

plt.close('all')

fig = plt.figure()
plt.bar(months_df, facecream_df, width = -0.4, align = "edge",
        color = "blue", label = "facecream")
plt.bar(months_df, facewash_df, width = 0.4, align = "edge",
        color = "orange", label = "facewash")
plt.legend()
plt.title("facewash and facecram sales data")
plt.xlabel("month number")
plt.ylabel("sales units in number")
plt.grid(True)