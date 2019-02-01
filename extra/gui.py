# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:07:09 2019

@author: p0p
"""

import tkinter as tk  # note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
from tkinter import *
# Code to add widgets will go here...
from PIL import Image, ImageTk

import os
import shutil
from os import path
from functools import partial
import glob
#%%
def renamer(filepath,file, rename):
    kappa = filepath+"\\{}".format(file)
    if path.exists(kappa):
        print('yes' + rename)
    	# get the path to the file in the current directory
        src = path.realpath(kappa);
    	# rename the original file
        os.rename(filepath+"\\{}".format(file),filepath+"\\{}".format(rename))


#%%
filepath = "C:\\Users\\p0p\\Desktop\\excel_drive"
os.chdir(filepath)
files = []

for file in glob.glob("*.jpeg"):
    files.append(file)
    
 
ii = -1
window = Tk()
labelText = StringVar()

window.title("rename file to")
 



 
def rename_clicked():
    new_name = codetxt.get()+ " " + qtytxt.get()+ " " + ratetxt.get()+ " " +' .jpeg'
    res = "file " +files[ii]+ " renamed to : " + new_name
    renamer(filepath,files[ii],new_name)
    lbl.configure(text= res)
    next_clicked()
    
def next_clicked():
    global ii 
    if ii < len(files)-1 :
        ii = ii + 1
    else:
        ii = 0
    txtxs = files[ii]
    canvas_image(txtxs)
    updateDepositLabel(ii)



def updateDepositLabel(ii):
    labelText.set(files[ii])
window.geometry("1000x800")
#Left Frameand its contents
leftFrame = Frame(window, width=600, height = 600)
leftFrame.grid(row=0, column=0, padx=10, pady=2)

w = Canvas(leftFrame, width=600, height=700)
def canvas_image(txtxs):
    image = Image.open(filepath+"\\{}".format(txtxs))
    image =  image.resize((900,700), Image.ANTIALIAS) 
    photo = ImageTk.PhotoImage(image)
    w.image = photo
    w.create_image(300,0, image=w.image, anchor='n')
next_btn = Button(leftFrame, text="Next",command=next_clicked)

w.grid(row=0, column=0, padx=10, pady=2)
next_btn.grid(row=2, column=0, padx=10, pady=2)

#Right Frame and its contents
rightFrame = Frame(window, width=600, height = 600)
rightFrame.grid(row=0, column=1, padx=10, pady=2)
lbl = Label(rightFrame, textvariable=labelText)
rename_btn = Button(rightFrame, text="Rename", command=rename_clicked)


#'CODE'
codelbl = Label(rightFrame, text='code')
codetxt = Entry(rightFrame,width=10)
#'QUANTITY'
qtylbl = Label(rightFrame, text='qty')
qtytxt = Entry(rightFrame,width=10)
#'RATE'
ratelbl = Label(rightFrame, text='rate')
ratetxt = Entry(rightFrame,width=10)
#'PICTURE'
#'DESCRIPTION'
lbl.grid(row=0, column=0, padx=10, pady=2)
codelbl.grid(row=1, column=0, padx=10, pady=2)
codetxt.grid(row=1, column=1, padx=10, pady=2)
qtytxt.grid(row=2, column=1, padx=10, pady=2)
qtylbl.grid(row=2, column=0, padx=10, pady=2)
ratelbl.grid(row=3, column=0, padx=10, pady=2)
ratetxt.grid(row=3, column=1, padx=10, pady=2)
rename_btn.grid(row=4, column=1, padx=10, pady=2)

window.mainloop()
#%%
x = 0
def k():
    global x
    x = x+2
    
print(x)
k()
print(x)