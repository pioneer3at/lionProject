import ttkbootstrap as ttk
from tkinter import filedialog, messagebox, Button, DISABLED, NORMAL, Entry, StringVar
from tkinter.messagebox import showerror, askyesno
from tkinter import colorchooser
from PIL import Image, ImageOps, ImageTk, ImageFilter, ImageGrab
import os
# from gsCamera import GsCamera
from orinHQCamera import OrinHQCamera
# from piHQCamera import PiHQCamera
import time
import cv2
import json 
import copy
from config import *
# from videoProcessingGPU import VideoProcessingGPU
from videoProcessing import VideoProcessing
import shutil
from config import *
import numpy as np
# import Jetson.GPIO as GPIO

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
PROGRAM_DIR = os.path.join(MAIN_DIR, 'programs/')
if not os.path.exists(PROGRAM_DIR):
    os.makedirs(PROGRAM_DIR)

class LionEditor():
    def __init__(self):
        self.root = ttk.Window(themename="cosmo")
        self.root.title("LION Training Tool")
        self.root.attributes('-fullscreen', False)  # Set fullscreen mode
        width_ = self.root.winfo_screenwidth()-100  # Get the screen width
        height_ = self.root.winfo_screenheight()-100  # Get the screen height
        self.root.geometry(f"{width_}x{height_}")  # Set window geometry to fullscreen
        # self.root.geometry("1000x900+300+110")
        # self.root.resizable(0, 0)
        icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/lion.png'))
        self.root.iconphoto(False, icon)

        # the left frame to contain the 4 buttons
        self.left_frame = ttk.Frame(self.root, width=300, height=600)
        self.left_frame.pack(side="left", fill="y")

        # the left frame to contain the 4 buttons
        self.bot_frame = ttk.Frame(self.root, width=1200, height=200)
        self.bot_frame.pack(side="bottom", fill="y")

        # the left frame to contain the 4 buttons
        self.info_frame = ttk.Frame(self.root, width=300, height=600)
        self.info_frame.pack(side="left", fill="y")
        ttk.Label(self.info_frame, text="Information", background="white").pack(side="top")

        # the right canvas for displaying the image
        self.canvas = ttk.Canvas(self.root, width=WIDTH, height=HEIGHT)
        self.canvas.pack()

        # label
        self.info_label = ttk.Label(self.info_frame, text="", background="white")
        self.info_label.pack(padx=0, pady=2)

        self.xy_label = ttk.Label(self.bot_frame, background="white")
        self.xy_label.pack(side="top", padx=0, pady=2)

        camera_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/camera.png')).subsample(12, 12)
        capture_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/capture.png')).subsample(12, 12)
        record_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/record.png')).subsample(12, 12)
        image_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/image.png')).subsample(12, 12)
        add_program_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/addProgram.png')).subsample(12, 12)
        load_program_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/loadProgram.png')).subsample(12, 12)
        run_program_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/run.png')).subsample(12, 12)
        review_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/review.png')).subsample(12, 12)
        save_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/diskette.png')).subsample(12, 12)
        exit_icon = ttk.PhotoImage(file=os.path.join(MAIN_DIR, 'icons/exit.png')).subsample(12, 12)
        
        camera_button = ttk.Button(self.left_frame, image=camera_icon, bootstyle="light", command=self.open_camera)
        camera_button.pack(pady=5)
        self.camera_label = ttk.Label(self.left_frame, text="Open Camera", background="white")
        self.camera_label.pack(padx=0, pady=2)

        self.capture_button = ttk.Button(self.left_frame, image=capture_icon, bootstyle="light", command=self.capture_image)
        self.capture_button.pack(pady=5)
        ttk.Label(self.left_frame, text="Capture Image", background="white").pack(padx=0, pady=2)
        self.capture_button.config(state=DISABLED)

        self.record_button = ttk.Button(self.left_frame, image=record_icon, bootstyle="light", command=self.record_video)
        self.record_button.pack(pady=5)
        self.record_label = ttk.Label(self.left_frame, text="Record Video", background="white")
        self.record_label.pack(padx=0, pady=2)
        self.record_button.config(state=DISABLED)

        image_button = ttk.Button(self.left_frame, image=image_icon, bootstyle="light", command=self.load_image)
        image_button.pack(pady=5)
        ttk.Label(self.left_frame, text="Load Image", background="white").pack(padx=0, pady=2)

        self.add_program_button = ttk.Button(self.left_frame, image=add_program_icon, bootstyle="light", command=self.add_program)
        self.add_program_button.pack(pady=5)
        ttk.Label(self.left_frame, text="Create Program", background="white").pack(padx=0, pady=2)
        self.add_program_button.config(state=DISABLED)

        load_program_button = ttk.Button(self.left_frame, image=load_program_icon, bootstyle="light", command=self.load_program)
        load_program_button.pack(pady=5)
        ttk.Label(self.left_frame, text="Load Program", background="white").pack(padx=0, pady=2)

        self.run_program_button = ttk.Button(self.left_frame, image=run_program_icon, bootstyle="light", command=self.run_program)
        self.run_program_button.pack(pady=5)
        self.program_label = ttk.Label(self.left_frame, text="Run Program", background="white")
        self.program_label.pack(padx=0, pady=2)
        self.run_program_button.config(state=DISABLED)

        self.review_button = ttk.Button(self.left_frame, image=review_icon, bootstyle="light", command=self.review)
        self.review_button.pack(pady=5)
        ttk.Label(self.left_frame, text="Review Program", background="white").pack(padx=0, pady=2)
        self.review_button.config(state=DISABLED)

        self.save_button = ttk.Button(self.left_frame, image=save_icon, bootstyle="light", command=self.save_program)
        self.save_button.pack(pady=5)
        ttk.Label(self.left_frame, text="Save", background="white").pack(padx=0, pady=2)
        self.save_button.config(state=DISABLED)

        self.exit_button = ttk.Button(self.left_frame, image=exit_icon, bootstyle="light", command=self.on_closing)
        self.exit_button.pack(pady=5)
        ttk.Label(self.left_frame, text="Exit", background="white").pack(padx=0, pady=2)

        self.rect = None
        self.file_path = None
        self.image_available = False

        self.start_x = None
        self.start_y = None
        self.x = self.y = 0

        self.process_video = False
        self.record_video_flag = False
        self.camera_openning = False
        self.multiple_areas_choosing = False
        self.program_name = None
        self.output_video = None
        self.recording_duration_label = None
        self.recording_start_time = None
        self.step_frames = []
        self.info_content = None
        self.vp = None

        self.current_widget = None
        self.canvas.bind('<Motion>', self.on_motion)    

        self.mouse_info = [0, 0, 0, 0, 0] # x, y, width, height, area

        self.camera = OrinHQCamera(resolution=(WIDTH, HEIGHT),framerate=FR)
        self.image = None

        self.program_config = copy.deepcopy(PROGRAM_CONFIG_TEMPLATE)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    # def init_gpio(self):
    #     GPIO.setmode(GPIO.BOARD)
    #     GPIO.setwarnings(False)

    # def warn_gpio(self):
    #     pass

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Would you like to quit?"):
            print("Destroying root ...")
            self.root.destroy()
            print("Stopping camera ...")
            self.camera.stop()

    def open_camera(self):
        if self.camera_openning == False:
            self.camera.start()
            time.sleep(2)
            self.stream_camera()
            self.capture_button.config(state=NORMAL)
            self.record_button.config(state=NORMAL)
            self.camera_label.config(text="Close Camera")
        elif self.camera_openning == True:
            self.root.after_cancel(self.stream)
            self.camera.stop_thread()
            self.capture_button.config(state=DISABLED)
            self.record_button.config(state=DISABLED)
            self.camera_label.config(text="Open Camera")
            self.process_video = False

        self.camera_openning = not self.camera_openning

    def show_result(self):
        if self.vp != None:
            if self.vp.total_quantity == 0:
                _percentage = 0
            elif self.vp.total_quantity > 0:
                _percentage = int(100*self.vp.correct_quantity/self.vp.total_quantity)
            self.info_label.config(text="Total Quantity: {} \nCorrect Quantity: {}\nYield: {} (%)".format(self.vp.total_quantity, self.vp.correct_quantity, _percentage))

    def stream_camera(self):
        try:
            if self.record_video_flag:
                self.canvas.create_text(300, 100, text="Recording Video. Turned off preview", font=("Arial", 12), fill="black")
                self.video_duration_label.config(text="Duration: {} s".format(round(time.time() - self.recording_start_time,2)))
            else:
                self.image = self.camera.read()
                if self.process_video:
                    start_time = time.time()
                    if self.load_prog['typeId'] == 'A':
                        self.image, ret, percent = self.vp.programA(self.image)
                        time_consumed   = time.time() - start_time  
                        self.fps             = 1/time_consumed

                        DETECTION_LIST.insert(len(DETECTION_LIST) - 1, DETECTION_LIST.pop(0))
                        if ret == True:
                            DETECTION_LIST[-1] = True
                        else:
                            DETECTION_LIST[-1] = False

                        print("FPS: {} | Last {}: {}".format(round(self.fps, 2), TOTAL_QUAN_FOR_JUDGEMENT, DETECTION_LIST.count(True)))
                        if DETECTION_LIST.count(True) >= ACCEPTANCE_VALUE:
                            color = (0, 255, 0)
                            EVER_DETECTED = True
                            print('\x1b[6;30;42m' + 'YES!' + '\x1b[0m')

                        else:
                            print('\x1b[6;33;41m' + 'No!' + '\x1b[0m')

                            color = (255,0,0)

                    elif self.load_prog['typeId'] == 'UPG':
                        self.image = self.vp.programUPG(self.image)
                        time_consumed   = time.time() - start_time  
                        self.fps             = 1/time_consumed

                        print("FPS: {}".format(round(self.fps, 2)))
                    
                    elif self.load_prog['typeId'] == 'B':
                        self.image = self.vp.programB(self.image)[0]
                        time_consumed   = time.time() - start_time  
                        self.fps             = 1/time_consumed

                        print("FPS: {}".format(round(self.fps, 2)))
                        self.canvas.create_text(300, 100, text="FPS: {}".format(round(self.fps, 2)), fill="black", font=("Arial", 12))

                    elif self.load_prog['typeId'] == 'C':
                        self.image = self.vp.programC(self.image)[0]
                        time_consumed   = time.time() - start_time  
                        self.fps             = 1/time_consumed

                        print("FPS: {}".format(round(self.fps, 2)))
                        self.canvas.create_text(300, 100, text="FPS: {}".format(round(self.fps, 2)), fill="black", font=("Arial", 12))

                    elif self.load_prog['typeId'] == 'D':
                        self.image = self.vp.programD(self.image)
                        time_consumed   = time.time() - start_time  
                        self.fps             = 1/time_consumed

                        print("FPS: {}".format(round(self.fps, 2)))
                        self.canvas.create_text(300, 100, text="Background Subtraction | FPS: {}".format(round(self.fps, 2)), fill="black", font=("Arial", 12))

                    elif self.load_prog['typeId'] == 'E':
                        result      = self.vp.programE(self.image)
                        self.image  = result[0]
                        response    = result[2]
                        time_consumed   = time.time() - start_time  
                        self.fps             = int(1/time_consumed)

                        print("FPS: {}".format(round(self.fps, 2)))
                        self.canvas.create_text(300, 100, text="Background Subtraction | FPS: {}".format(round(self.fps, 2)), fill="black", font=("Arial", 12))

                self.tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            
            self.show_result()
            
            self.stream = self.root.after(int(1000/FR), self.stream_camera)  # <== start the repeating process
        
        except Exception as e:
            self.canvas.create_text(300, 100, text="Cannot open Camera, please check Camera again or reboot", font=("Arial", 12), fill="black")
            self.capture_button.config(state=DISABLED)
            print("Exception during open camera: ", e)

    def capture_image(self):
        self.root.after_cancel(self.stream)
        self.image_available = True
        self.add_program_button.config(state=NORMAL)
        self.open_camera()

    def record_video(self):
        if self.record_video_flag == False:
            self.record_label.config(text="Stop recording")

            self.video_duration_label = ttk.Label(self.bot_frame, text="Video Duration", background="white")
            self.video_duration_label.pack(padx=0, pady=2)
            self.video_name_label = ttk.Label(self.bot_frame, text="Video Name", background="white")
            self.video_name_label.pack(padx=0, pady=2)
            self.video_name_entry = Entry(self.bot_frame, textvariable = StringVar())
            self.video_name_entry.pack(padx=0, pady=2)

            self.recording_start_time = time.time()

            self.camera.stop_thread()
            self.camera.record_video()

            self.record_video_flag = True

        elif self.record_video_flag == True:
            self.record_label.config(text="Record Video")
            directory = filedialog.askdirectory()
            if directory:
                filename = self.video_name_entry.get()
                if filename:
                    full_path = directory + '/' + filename + VIDEO_EXTENSION
                    shutil.move(DEFAULT_VIDEO_FILENAME, full_path)
                self.output_video = None

                self.video_duration_label.destroy()
                self.video_name_label.destroy()
                self.video_name_entry.destroy()
                self.camera.stop_recording_video()
                self.camera.start()
            
            self.record_video_flag = False

    def add_program(self):
        # Destroy button if already created
        self.destroy_program_objects()

        if self.program_name == None:
            ttk.Label(self.bot_frame, text="Set Name", background="white").pack(padx=0, pady=2)
            self.program_name = Entry(self.bot_frame, textvariable = StringVar())
            self.program_name.pack(padx=0, pady=2)

            self.select_program_label = ttk.Label(self.bot_frame, text="Select Program:", background="white")
            self.select_program_label.pack(padx=0, pady=2)

            # a list of filters
            program_list = [i['type'] for i in (self.program_config)]

            # combobox for the filters
            self.program_combobox = ttk.Combobox(self.bot_frame, values=program_list, width=30, justify='center')
            self.program_combobox.bind('<<ComboboxSelected>>', self.program_modified) 
            self.program_combobox.pack(padx=10, pady=5)

        self.add_program_button.config(state=DISABLED)
        self.save_button.config(state=NORMAL)

    def load_program(self):
        self.load_prog_dir = None
        self.load_prog_dir = filedialog.askdirectory(initialdir=PROGRAM_DIR, title="Choose Program Directory")
        if self.load_prog_dir != None:
            self.file_path = os.path.join(self.load_prog_dir, TRAIN_IMAGE_NAME)
            self.image = Image.open(self.file_path)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.image_available = True

            self.prog_data_file = os.path.join(self.load_prog_dir, DATA_INFO_FILENAME)
            self.load_prog = json.load(open(self.prog_data_file))
            for step in self.load_prog["steps"]:
                if type(step['data'][0]) != list: 
                    self.canvas.create_rectangle(step['data'][0], step['data'][1], step['data'][2], step['data'][3], outline=step['color'], tags="review")
                else:
                    for data in step['data']:
                        self.canvas.create_rectangle(data[0], data[1], data[2], data[3], outline=step['color'], tags="review")
                if step['name'] == 'Choose convey area':
                    self.canvas.create_line(0, step['data'][1], WIDTH, step['data'][1], fill=step['color'], width=5, tags="review")
                    self.canvas.create_line(0, step['data'][3], WIDTH, step['data'][3], fill=step['color'], width=5, tags="review")
            
            self.run_program_button.config(state=NORMAL)

    def run_program(self):
        self.vp = VideoProcessing(True, self.load_prog_dir)
        if self.process_video:
            self.program_label.config(text="Run Program")
        else:
            self.program_label.config(text="Stop Program")
        self.process_video = not self.process_video
        self.erase_review_rectangle()

    def load_image(self):
        self.file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image Files", "*.jpg")])
        if self.file_path:
            self.image = Image.open(self.file_path)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.image_available = True
    
    def review(self):
        cur_prog_name = self.program_combobox.get()
        for prog in self.program_config:
            if prog['type'] == cur_prog_name:
                for step in prog["steps"]:
                    if type(step['data'][0]) != list: 
                        self.canvas.create_rectangle(step['data'][0], step['data'][1], step['data'][2], step['data'][3], outline=step['color'], tags="review")
                    else:
                        for data in step['data']:
                            self.canvas.create_rectangle(data[0], data[1], data[2], data[3], outline=step['color'], tags="review")
                    if step['name'] == 'Choose convey area':
                        self.canvas.create_line(0, step['data'][1], WIDTH, step['data'][1], fill=step['color'], width=5, tags="review")
                        self.canvas.create_line(0, step['data'][3], WIDTH, step['data'][3], fill=step['color'], width=5, tags="review")

    def erase_review_rectangle(self):
        self.canvas.delete("review")

    def erase_data(self):
        msg_box = messagebox.askquestion('Warning!!!', 'Are you sure?', icon='warning')
        if msg_box == 'yes':
            self.canvas.delete("review")

            cur_prog_name = self.program_combobox.get()
            for prog in self.program_config:
                if prog['type'] == cur_prog_name:
                    for step in prog["steps"]:
                        step['data'] = [0, 0, 0, 0]

            self.review_button.config(state=DISABLED)
            # self.erase_button.config(state=DISABLED)
        else:
            pass

    def save_program(self):
        self.add_program_button.config(state=NORMAL)

        cur_prog_type = self.program_combobox.get()
        cur_prog_name = self.program_name.get()
        prog_data = None
        for prog in self.program_config:
            if prog['type'] == cur_prog_type:
                prog['name'] = self.program_name.get()
                prog_data = prog
        invalid_step_id_list = []
        for step in prog_data['steps']:
            if not step['valid']:
                invalid_step_id_list.append(step['stepId'])
        for entry in prog_data['entries']:
            entry['data'] = entry["object"].get()
            try:
                if int(entry['data']) > 0: entry['valid'] = True
                else: entry['valid'] = False
            except: entry['valid'] = False
            if not entry['valid']:
                messagebox.showerror('Error', str(entry["name"]) + ' is invalid')
                return
        for item in prog_data['combobox']:
            item['data'] = item["object"].get()

        prog_list = [ f.name for f in os.scandir(PROGRAM_DIR) if f.is_dir() ]
        if cur_prog_name == '':
            messagebox.showerror('Error', 'Program name is empty')
        else:
            cur_prog_dir = os.path.join(PROGRAM_DIR, cur_prog_name)
            if cur_prog_name in prog_list:
                msg_box = messagebox.askquestion('Warning!!!', 'Program name existed. Overwrite?', icon='warning')
                if msg_box == 'no': 
                    return
                else:
                    shutil.rmtree(cur_prog_dir, ignore_errors=False, onerror=None)

            if invalid_step_id_list == []:
                cur_prog_template_dir = os.path.join(cur_prog_dir, 'templates')
                for dir in [cur_prog_dir, cur_prog_template_dir]:
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                for step in prog_data['steps']:
                    if step['saveImage']:
                        cropped_image = self.image[step['data'][1]:step['data'][3], step['data'][0]:step['data'][2]]
                        cv2.imwrite(os.path.join(cur_prog_template_dir, step['imageName']), cropped_image)

                # Remove button object
                prog_to_write = {}
                for key in prog_data:
                    if key == "steps":
                        prog_to_write[key] = []
                        for step in prog_data[key]:
                            _step = {}
                            for item in step:
                                if item != "button":
                                    _step[item] = step[item]
                            prog_to_write[key].append(_step)
                    elif key == "entries":
                        prog_to_write[key] = []
                        for e in prog_data[key]:
                            _entry = {}
                            for k in e:
                                if k not in ["object", "labelObject"]:
                                    _entry[k] = e[k]
                            prog_to_write[key].append(_entry)
                    elif key == "combobox":
                        prog_to_write[key] = []
                        for e in prog_data[key]:
                            _entry = {}
                            for k in e:
                                if k not in ["object", "labelObject"]:
                                    _entry[k] = e[k]
                            prog_to_write[key].append(_entry)
                    else:
                        prog_to_write[key] = prog_data[key]

                # for step in prog_data['steps']:
                #     del step['button']
                with open(os.path.join(cur_prog_dir, DATA_INFO_FILENAME), 'w') as convert_file: 
                    convert_file.write(json.dumps(prog_to_write, indent=4))
                
                cv2.imwrite(os.path.join(cur_prog_dir, TRAIN_IMAGE_NAME), self.image)
                print("Saved!!!")
                self.info_label.config(text="Saved Program!")

            else:
                messagebox.showerror('Error', 'Invalid steps: {}'.format(invalid_step_id_list))

    def destroy_program_objects(self):
        # Destroy button if already created
        for prog in self.program_config:
            for step in prog["steps"]:
                if step['button']:
                    step['button'].destroy()
            for entry in prog["entries"]:
                if entry['object']:
                    entry['labelObject'].destroy()
                    entry['object'].destroy()
            if "combobox" in prog:
                for item in prog["combobox"]:
                    if entry['object']:
                        item['labelObject'].destroy()
                        item['object'].destroy()
        for frame in self.step_frames:
            frame.destroy()

    def program_modified(self, event):
        cur_prog_name = self.program_combobox.get()
        self.destroy_program_objects()
        self.program_config = copy.deepcopy(PROGRAM_CONFIG_TEMPLATE)

        for prog in self.program_config:
            if prog['type'] == cur_prog_name:
                self.step_frames = [None] * (int(prog["packQuantity"]) + 1)
                for i in range(len(self.step_frames)):
                    self.step_frames[i] = ttk.Frame(self.bot_frame, width=int(WIDTH/(int(prog["packQuantity"]) + 1)), height=600)
                    self.step_frames[i].pack(side="left", fill="y")
                if prog['entries'] != []:
                    for item in prog['entries']:
                        item['labelObject'] = ttk.Label(self.step_frames[item["packId"]], text=item['name'], background="white")
                        item['labelObject'].pack(padx=0, pady=2)
                        item['object'] = Entry(self.step_frames[item["packId"]], textvariable = StringVar())
                        item['object'].pack(padx=0, pady=2)
                        item['object'].insert(0, str(item['data']))
                if "combobox" in prog:
                    for item in prog["combobox"]:
                        item['labelObject'] = ttk.Label(self.step_frames[item["packId"]], text=item['name'], background="white")
                        item['labelObject'].pack(padx=0, pady=2)
                        item['object'] = ttk.Combobox(self.step_frames[item["packId"]], values=item['options'], width=30, justify='center')
                        item['object'].pack(padx=10, pady=5)
                        item['object'].current(0)
                for step in prog["steps"]:
                    if step['multiple']:
                        step['button'] = Button(self.step_frames[step["packId"]], text=str(step['stepId'])+'. '+step['name'])
                        step['button'].config(fg='black', bg=step['color'], command=lambda obj=step['button']: self.choose_multiple_areas(obj))
                        step['button'].pack(side="top", padx=5)
                    else:
                        step['button'] = Button(self.step_frames[step["packId"]], text=str(step['stepId'])+'. '+step['name'])
                        step['button'].config(fg='black', bg=step['color'], command=lambda obj=step['button']: self.set_current_step(obj))
                        step['button'].pack(side="top", padx=5)

    def choose_multiple_areas(self, widget):
        cur_prog_name = self.program_combobox.get()
        for prog in self.program_config:
            if prog['type'] == cur_prog_name:
                for step in prog["steps"]:
                    if step['button'] == widget:
                        for entry in prog['entries']:
                            if entry['name'] == step['entryName']:
                                step['quantity'] = entry['object'].get()
                                self.collected_areas_list = []
                                self.multiple_areas_choosing = True
                                self.collecting_quantity = int(step['quantity'])
                                if self.collecting_quantity > 0:
                                    self.collecting_step = step
                                    self.set_current_step(self.collecting_step['button'])
                                    self.collecting_areas()
                                    print("Choosing {} areas".format(self.collecting_quantity))
                                else:
                                    print('Quantity have to be bigger than 0')
    
    def collecting_areas(self):
        if self.collecting_step['valid']:
            self.collected_areas_list.append(self.collecting_step['data'])
            self.collecting_step['valid'] = False

        if len (self.collected_areas_list) == self.collecting_quantity:
            self.multiple_areas_choosing = False
            self.root.after_cancel(self.wait_for_collecting_areas)
            self.collecting_step['data'] = self.collected_areas_list
            self.unbind_rectangle()
            self.collecting_step['valid'] = True
            print("Finished collecting {} areas".format(self.collecting_quantity))

        else:
            self.wait_for_collecting_areas = self.root.after(100, self.collecting_areas)  # <== start the repeating process

    def set_current_step(self, widget):
        self.current_widget = widget
        self.erase_rects()
        self.rectangle_draw()
        
        self.erase_review_rectangle()

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        for prog in self.program_config:
            for step in prog["steps"]:
                if step['button'] == self.current_widget:
                    color = step['color']

        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline=color, tags="rectangle")

    def on_move_press(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)

        self.mouse_info[2] = int(curX - self.start_x)
        self.mouse_info[3] = int(curY - self.start_y)
        self.mouse_info[4] = int(self.mouse_info[2]*self.mouse_info[3])
        self.xy_label.config(text= 'x=' + str(int(self.mouse_info[0])) + ', ' + 'y=' + str(int(self.mouse_info[1])) + ', ' + 'w=' + str(int(self.mouse_info[2]))+ ', ' + 'h=' + str(int(self.mouse_info[3]))+ ', ' + 'area=' + str(int(self.mouse_info[4])))

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)  

    def on_button_release(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y) 

        msg_box = messagebox.askquestion('Warning!!!', 'Save the area?', icon='warning')
        if not self.multiple_areas_choosing:
            self.unbind_rectangle()
        
        if msg_box == 'yes':
            for prog in self.program_config:
                for step in prog["steps"]:
                    if step['button'] == self.current_widget:
                        step['valid'] = True
                        if 'collect' in step:
                            if step['collect'] == 'hsv':
                                print("Collecting hsv values")
                                cropped_image = self.image[int(self.start_y):int(self.end_y), int(self.start_x): int(self.end_x)]
                                if cropped_image.any():
                                    hsv_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
                                    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

                                step['data'] = [int(self.start_x), int(self.start_y), int(self.end_x), int(self.end_y), int(np.min(h)), int(np.max(h)), int(np.min(s)), int(np.max(s)), int(np.min(v)), int(np.max(v))]
                        else:
                            step['data'] = [int(self.start_x), int(self.start_y), int(self.end_x), int(self.end_y)]
            
            if not self.multiple_areas_choosing:
                self.erase_rects()
                self.review_button.config(state=NORMAL)
                # self.erase_button.config(state=NORMAL)
        else:
            if not self.multiple_areas_choosing:
                self.erase_rects()
    
    def unbind_rectangle(self):
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def on_motion(self, event):
        if self.image_available:
            self.mouse_info[0], self.mouse_info[1] = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
            self.xy_label.config(text= 'x=' + str(int(self.mouse_info[0])) + ', y=' + str(int(self.mouse_info[1])) + ', w=' + str(int(self.mouse_info[2]))+ ', h=' + str(int(self.mouse_info[3]))+ ', area=' + str(int(self.mouse_info[4])))

    def erase_rects(self):
        self.mouse_info[2] = 0
        self.mouse_info[3] = 0
        self.mouse_info[4] = 0
        self.canvas.delete("rectangle")
        self.rect = None

    def rectangle_draw(self):
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

if __name__ == '__main__':
    LionEditor()
