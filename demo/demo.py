import winsound
import sys
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import os, sys, subprocess
import numpy as np
import time
import tkinter.font as tf
from tkinter.filedialog import askdirectory , askopenfilename

detect = None
output = None

dic = {0:'initialize....'}

string = '--------------------------------'

def update_clock():

    global string
    global label
    global button
    global dic
    global window
    global people

    now = time.time()

    now = now - begin
    
    i = int((now)*10)
    
    
    if detect[i] == True:
        if output[i] not in dic:
            dic [ output[i] ]  = 'people   ' + chr(people)
            people+=1
        string = dic [ output[i] ]
    else :
        string = 'no voice'
    
    if i<10:
        string = 'initialize....'
    
    
    string = ('%.1f   s' %now)+ '      ' + string
    

    label.configure(text=string)

    window.after(30, update_clock)
    

def click():

    global people

    global begin

    global button

    global detect

    global output

    people = 65

    wav_name = path1.get().split('/')[-1]

    detect = np.load(wav_name.split('.')[0]+'detect.npy')

    output = np.load(wav_name.split('.')[0]+'output.npy')

    begin = time.time()

    winsound.PlaySound(wav_name, winsound.SND_ALIAS | winsound.SND_ASYNC)

    button.config(state="disabled")

    window.after(1, update_clock)
    

def selectPath1():
    
    path_ = askopenfilename(initialdir =os.getcwd(),filetypes = (("wav", "*.wav"), ("All files", "*")))
    path1.set(path_)
    button.config(state="active")



i = 0
people = 0
begin = 0
window = tk.Tk()
window.title('Demo')
window.geometry('800x600')
window.configure(background='white')

path1 = StringVar()
label1 = Label(window,text = 'Path : ').place(x=25, y=25,width = 100 , height = 30 )
entry1 = Entry(window, textvariable = path1,width=100).place(x=150, y=25 ,width = 400 , height = 30 )
button1 = Button(window, text = 'Enter', command = selectPath1).place(x=575, y=25,width = 100 , height = 30 )

label = Label(window,text = string, font = ('Arial',24) )
label.place( x= 200 , y=100 ,width = 400 , height = 100 )
button = Button(window,text = 'start',command=click,font = ('Arial',24))
button.place( x= 300 , y= 300 ,width = 200 , height = 100 )

window.mainloop()
