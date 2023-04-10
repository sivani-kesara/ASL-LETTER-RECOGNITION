from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from keras.models import model_from_json
import operator
import time
import sys
from spylls.hunspell import Dictionary
from string import ascii_uppercase
class Application:
    def __init__(self):
        self.hs = Dictionary.from_files('en_US')    
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open("model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("model-bw.h5")
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
          self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x1100")
        self.panel = tk.Label(self.root)
        self.panel.place(x = 135, y = 10, width = 640, height = 640)
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 460, y = 95, width = 310, height = 310)
        self.T = tk.Label(self.root)
        self.T.place(x=31,y = 17)
        self.T.config(text = "Sign Language to Text",font=("courier",40,"bold"))
        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 500,y=640)
        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10,y = 640)
        self.T1.config(text="Character :",font=("Courier",40,"bold"))
        """self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220,y=700)
        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 700)
        self.T2.config(text ="Word :",font=("Courier",40,"bold"))
        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 350,y=760)
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10,y = 760)
        self.T3.config(text ="Sentence :",font=("Courier",40,"bold"))
        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250,y = 820)
        self.T4.config(text = "Suggestions",fg="red",font = ("Courier",40,"bold"))"""
        self.btcall = tk.Button(self.root,command = self.action_call,height = 0,width = 0)
        self.btcall.config(text = "About",font = ("Courier",14))
        self.btcall.place(x = 825, y = 0)
        self.bt1=tk.Button(self.root, command=self.action1,height = 0,width = 0)
        self.bt1.place(x = 26,y=890)
        self.bt2=tk.Button(self.root, command=self.action2,height = 0,width = 0)
        self.bt2.place(x = 325,y=890)
        self.bt3=tk.Button(self.root, command=self.action3,height = 0,width = 0)
        self.bt3.place(x = 625,y=890)
        self.bt4=tk.Button(self.root, command=self.action4,height = 0,width = 0)
        self.bt4.place(x = 125,y=950)
        self.bt5=tk.Button(self.root, command=self.action5,height = 0,width = 0)
        self.bt5.place(x = 425,y=950)
        self.str=""
        self.word=""
        self.current_symbol="Empty"
        self.photo="Empty"
        self.video_loop()
    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1, y1, x2, y2 = int(0.5*frame.shape[1]), 10, frame.shape[1]-10, int(0.5*frame.shape[1])
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            self.panel.imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.config(image=self.panel.imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            self.predict(res)
            self.current_image2 = Image.fromarray(res)
            self.panel2.imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.config(image=self.panel2.imgtk)
            self.panel3.config(text=self.current_symbol, font=("Courier", 50))
            """self.panel4.config(text=self.word, font=("Courier", 40))
            self.panel5.config(text=self.str, font=("Courier", 40))"""
            predicts = self.hs.suggest(self.word)
            predicts = list(predicts)
            for i, predict in enumerate(predicts[:5]):
                getattr(self, f"bt{i+1}").config(text=predict, font=("Courier", 20)) if len(predicts) > i else getattr(self, f"bt{i+1}").config(text="")
            self.root.after(30, self.video_loop)
    def predict(self,test_image):
        test_image = cv2.resize(test_image, (128,128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        prediction={}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
    def action1(self):
        predicts=self.hs.suggest(self.word)
        if(len(predicts) > 0):
            self.word=""
            self.str+=" "
            self.str+=predicts[0]
    def action2(self):
        predicts=self.hs.suggest(self.word)
        if(len(predicts) > 1):
            self.word=""
            self.str+=" "
            self.str+=predicts[1]
    def action3(self):
        predicts=self.hs.suggest(self.word)
        if(len(predicts) > 2):
            self.word=""
            self.str+=" "
            self.str+=predicts[2]
    def action4(self):
        predicts=self.hs.suggest(self.word)
        if(len(predicts) > 3):
            self.word=""
            self.str+=" "
            self.str+=predicts[3]
    def action5(self):
        predicts=self.hs.suggest(self.word)
        if(len(predicts) > 4):
            self.word=""
            self.str+=" "
            self.str+=predicts[4]
    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()
    def action_call(self):
        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("900x900")
        names = ["Sivani", "Akshith", "Chandana"]
        numbers = ["221910320027", "221910320026", "221910320014"]
        images = ["27.png", "26.png", "14.png"]
        for i in range(3):
            x_pos = 100 + i * 250
            img = tk.PhotoImage(file=images[i])
            label_img = tk.Label(self.root1, image=img)
            label_img.image = img  # Prevent image from being garbage collected
            label_img.place(x=x_pos, y=50)
            label_name = tk.Label(self.root1, text=names[i], font=("Courier", 15, "bold"))
            label_name.place(x=x_pos, y=220)
            label_number = tk.Label(self.root1, text=numbers[i], font=("Courier", 15, "bold"))
            label_number.place(x=x_pos, y=270)
print("Starting Application...")
pba = Application()
pba.root.mainloop()