#*************************************#
#    EyelidNet                        #
# Vitay Lerner 2022                   #
# Hebrew University of Jerusalem      #
# UI of manual annotation of points   #
#*************************************#
import tkinter as tk
from numpy import *
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import image


from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
K=['imgn']+['x{}'.format(i) for i in range(8)]+ ['y{}'.format(i) for i in range(8)]


class EyelidNet_Annotation(tk.Tk):
    D=pd.DataFrame(columns=K)
    folder=""
    axState=None
    curImg=0
    #*************************************************#
    #         Working with paths                      #
    #*************************************************#
    def browse(self):
        # Opens "Selet Folder Dialog" 
        file_path = tk.filedialog.askdirectory()
        if not (file_path[-1]=='/'):
            file_path=file_path+'/'
        self.txtPath.delete(0,tk.END)
        self.txtPath.insert(0,file_path)
        #self.open_dir()
        
    def newSession(self):
        # Opens "Create New File" dialog
        file_path=tk.filedialog.asksaveasfilename(filetypes=[('CSV','*.csv')],
                                title='Create new session...',
                                defaultextension=".csv")
        try:
            self.txtSessionPath.delete(0,tk.END)
            self.txtSessionPath.insert(0,file_path)
            D=pd.DataFrame(columns=K)
            D.to_csv(file_path,index=False)
        except FileNotFoundError:
            self.txtSessionPath.delete(0,tk.END)
            self.txtSessionPath.insert(0,'Load or create a new session file')
    def browseSession(self):
        #
        file_path = tk.filedialog.askopenfilename(filetypes=[('CSV','*.csv')],
        title='Select previously created session...',)
        try:
            self.txtSessionPath.delete(0,tk.END)
            self.txtSessionPath.insert(0,file_path)
            self.D=pd.read_csv(file_path)
            self.curImg=self.get_first_available()
            self.lblCurImg.config(text='{}'.format(self.curImg))
            
            self.plotState()
            print (self.D.head())
        except IndexError:
            self.txtSessionPath.delete(0,tk.END)
            self.txtSessionPath.insert(0,'Load or create a new session file')
    #*************************************************#
    #         State of the session                    #
    #*************************************************# 
    def get_first_available(self):
        # get first not-yet-marked image number
        I=set(list(self.D['imgn']))
        N=set(range(self.files_count()))
        return sorted(list(N^I))[0]
    def files_count(self):
        fld=self.txtPath.get()
        if fld[-1]=='/':
            pass
        else:
            fld=fld+'/'
        try:
            onlyfiles = [f for f in os.listdir(fld) if os.path.isfile(os.path.join(fld, f))]
            return len(onlyfiles)
        except FileNotFoundError:
            self.txtPath.delete(0)
            self.txtPath.insert(0,'Please select folder with images')
            
            return 0
      
    def plotState(self):
        # Repots current situation with marking
        # in a form of table of numbers
        # green: done, red: not yet
        imgn=self.curImg
        self.lblCurImg.config(text='{}'.format(imgn))
        nimg=self.files_count()
        #f=self.fig1
        self.axState.clear()
        
        rows=50
        cols=ceil(nimg/rows)
        d_imgn=list(self.D['imgn'])
        for imgn in range(nimg):
            y=floor(imgn/cols)
            x=imgn%cols
            if imgn in d_imgn:
                clr=[0.05,0.8,0.05,0.95]
            else:
                clr =[0.8,0.05,0.05,0.2]
            self.axState.text(x,y,'{}'.format(imgn),color=clr,size=8)
        self.axState.set_xlim([-1,cols+1])
        self.axState.set_ylim([rows+1,0])
        self.axState.axis('off')
        self.tkfig1.draw()

    def image_next(self):
        # not used, get_first_available is used instead
        self.curImg=self.get_first_available()
        self.lblCurImg.config(text='{}'.format(self.curImg))
        self.image_show()
        
    #*************************************************#
    #         Core functions                          #
    #*************************************************# 
    def image_show(self):
        # Shows a current image and waits for 8 points
        # Then shows the result in the main window
        imgn=self.curImg
        self.lblCurImg.config(text='{}'.format(imgn))
        ax=self.axImg
        ax.clear()
        fld=self.txtPath.get()
        fname=fld+'img{:03d}.png'.format(imgn)
        img = image.imread(fname)

        fTmp=plt.figure()
        mgr=plt.get_current_fig_manager()
        try:
        	mgr.window.setGeometry(50,100,650,700)
        except:
        	pass
        axTmp=fTmp.add_subplot(111)
        
        plt.imshow(img)
        plt.title('Image #{}'.format(imgn))
        xy=fTmp.ginput(8,0)
        x=[int(xxy[0]) for xxy in xy]
        y=[int(xxy[1]) for xxy in xy]
        self.temp_x=x
        self.temp_y=y
        ax.imshow(img)
        ax.plot(x,y,'.-r')
        ax.axis('off')
        
        ax.set_title('Image #{} (result)'.format(imgn))
        self.tkFigImg.draw()
        
        plt.close(fTmp)
        
        x=self.temp_x
        y=self.temp_y
        d={'imgn':self.curImg}
        for i in range(8):
            d['x{}'.format(i)]=x[i]
            d['y{}'.format(i)]=y[i]
        self.D=self.D.append(d, ignore_index=True)
        self.plotState()
        self.D.to_csv(self.txtSessionPath.get(),index=False)
        
    def __init__(self):
        # Creates Main Window and UI assigns UI functions to 
        # events
        super().__init__()
        self.geometry('800x720+20+20')
        self.configure(bg='white')
        self.title('EyelidNet')


        self.fig1 = plt.Figure(figsize=(1.8, 6), dpi=100)
        self.axState=self.fig1.add_subplot(111)
        self.tkfig1 = FigureCanvasTkAgg(self.fig1, self)
        
        self.figImg=plt.Figure(figsize=(5.5,5.5),dpi=100)
        self.axImg=self.figImg.add_subplot(111)
        self.tkFigImg=FigureCanvasTkAgg(self.figImg, self)
        
        lblTitle=tk.Label(self,text='EyelidNet: Vitaly 2022')
        
        cmdPath=tk.Button(self,text='Images folder...',command=self.browse)
        cmdSessionLoad=tk.Button(self,text='Load Session...',command=self.browseSession)
        cmdSessionNew=tk.Button(self,text='New Session...',command=self.newSession)
        
  
        lblSessionPath=tk.Label(text='Session File:')
        self.txtPath=tk.Entry(self,width=60)
        self.txtSessionPath=tk.Entry(self,width=60)
        

        self.txtPath.insert(0,'Please select folder with images')
        self.txtSessionPath.insert(0,'Please load or create a new session file')
        
        button_quit = tk.Button(master=self, text="Quit", command=self.destroy)
        
        cmdStart=tk.Button(master=self, text="Work on current image", command=self.image_show)
        cmdNext=tk.Button(master=self, text="Work on next image", command=self.image_next)
        lblCurImgLabel=tk.Label(self,text='Current Image#')
        self.lblCurImg=tk.Label(self,text='N/A')
        
        lblTitle.place(x=300,y=10)
        lblTitle.config(font=("Courier", 16),fg="#000000",bg="white")


        cmdPath.place(x=20,y=50)
        self.txtPath.place(x=220,y=52)
        
        cmdSessionNew.place(x=20,y=90)
        cmdSessionLoad.place(x=120,y=90)
        self.txtSessionPath.place(x=220,y=92)
        
        cmdStart.place(x=20,y=120)
        cmdNext.place(x=190,y=120)
        lblCurImgLabel.place(x=350,y=120)
        self.lblCurImg.place(x=500,y=120)
        lblCurImgLabel.config(font=("Courier", 12),fg="#000000",bg="white")
        self.lblCurImg.config(font=("Courier", 12),fg="#000000",bg="white")
        
        self.tkfig1.get_tk_widget().place(x=610,y=20)
        self.tkFigImg.get_tk_widget().place(x=10,y=150)
        
        self.axState.axis('off')
        self.axImg.axis('off')
        self.axImg.text(0,10,'1. Select Folder with images')
        self.axImg.text(0,20,'2. Select or create new session')
        self.axImg.text(0,30,'3. Press Start')
        self.axImg.text(0,40,'4. Select 8 points')
        self.axImg.text(0,50,'5. If looks good, press next and return to 4\n       otherwise press start again')
        
        self.axImg.set_xlim([-5,50])
        self.axImg.set_ylim([100,-4])

        

if __name__ == '__main__':
    ENM = EyelidNet_Annotation()
    ENM.mainloop()
