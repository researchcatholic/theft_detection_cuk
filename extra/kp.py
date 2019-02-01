# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:17:51 2019

@author: p0p
"""

import openpyxl
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment
import os
import glob

#%%
filepath = "C:\\Users\\p0p\\Desktop\\excel_drive"
os.chdir(filepath)

fp = os.getcwd()  + '\\item_lists.xlsx'
wb = openpyxl.Workbook()
sheet = wb.active
a =  Alignment(horizontal='center', vertical = 'center', wrap_text=1)

#%%
sheet['A1'] = 'CODE'
sheet['B1'] = 'QUANTITY'
sheet['C1'] = 'RATE'
sheet['D1'] = 'PICTURE'
sheet['E1'] = 'DESCRIPTION'
descriptions = 'Generates random lorem ipsum text Lorem Ipsum Generator Lorem Ipsum Generator provides a GTK+ graphical user interface, a command-line interface'
#%%
i = sheet.max_row
for file in glob.glob("*.jpeg"):
    i = i + 1
    splits = file.split(' ')
    sheet['A{}'.format(i)] = splits[0]
    sheet['B{}'.format(i)] = splits[1]
    sheet['C{}'.format(i)] = splits[2].split('.')[0]
    img = Image(os.getcwd() + '\\'+file)
    img.height, img.width = 150,150
    sheet.add_image(img, 'D{}'.format(i))
    sheet.row_dimensions[i].height = 130
    sheet.column_dimensions['B'].width = 20
    sheet.column_dimensions['D'].width = 30
    sheet['E{}'.format(i)] = descriptions
    sheet.column_dimensions['E'].width = 50
#%%
for row_ in sheet:
    for cell_ in row_:
        cell_.alignment = a
#for col in sheet.columns:
#    max_length = 0
#    column = col[0].column # Get the column name
#    for cell in col:
#        if cell.column != 'D':
#            try: # Necessary to avoid error on empty cells
#                if len(str(cell.value)) > max_length:
#                    max_length = len(cell.value)
#            except:
#                pass
#adjusted_width = (max_length + 2) * 1.2
#sheet.column_dimensions[column].width = adjusted_width
       
#%%
wb.save(fp)
wb.close()