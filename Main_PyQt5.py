from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PyQt5 import QtCore, QtGui, QtWidgets

import tensorflow as tf
import facenet
import os
import sys
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import sqlite3
from datetime import datetime
import pyshine as ps
import subprocess

face_detector = cv2.CascadeClassifier('C:/Users/MyPC/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

def getProfile(id):
    conn = sqlite3.connect('C:/Users/MyPC/Downloads/FaceRecogniton/src/DataSet.db')
    query = "SELECT * FROM DataSet WHERE ID="+str(id)
    cursor = conn.execute(query)
    
    profile=None
    
    for row in cursor:
        profile = row
        
    conn.close()
    return profile

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.setFixedSize(986, 595)
        Dialog.setMouseTracking(False)
        Dialog.setTabletTracking(False)
        Dialog.setFocusPolicy(QtCore.Qt.NoFocus)
        Dialog.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setAutoFillBackground(False)
        Dialog.setStyleSheet("")
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(False)
        self.horizontalWidget = QtWidgets.QWidget(Dialog)
        self.horizontalWidget.setGeometry(QtCore.QRect(0, 0, 981, 581))
        self.horizontalWidget.setStyleSheet("")
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(0, -1, -1, -1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.horizontalWidget)
        self.label_3.setAutoFillBackground(False)
        self.label_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_3.setLineWidth(2)
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("Themnen.png"))
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bl_get = QtWidgets.QPushButton(self.horizontalWidget)
        self.bl_get.setStyleSheet("font: 11pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_get.setDefault(True)
        self.bl_get.setObjectName("bl_get")
        self.horizontalLayout_2.addWidget(self.bl_get)
        self.bl_cut = QtWidgets.QPushButton(self.horizontalWidget)
        self.bl_cut.setStyleSheet("font: 11pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_cut.setDefault(True)
        self.bl_cut.setObjectName("bl_cut")
        self.horizontalLayout_2.addWidget(self.bl_cut)
        self.bl_train = QtWidgets.QPushButton(self.horizontalWidget)
        self.bl_train.setStyleSheet("font: 11pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_train.setDefault(True)
        self.bl_train.setObjectName("bl_train")
        self.horizontalLayout_2.addWidget(self.bl_train)
        self.bl_cam = QtWidgets.QPushButton(self.horizontalWidget)
        self.bl_cam.setStyleSheet("font: 11pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_cam.setDefault(True)
        self.bl_cam.setObjectName("bl_cam")
        self.horizontalLayout_2.addWidget(self.bl_cam)
        self.bl_exit = QtWidgets.QPushButton(self.horizontalWidget)
        self.bl_exit.setStyleSheet("font: 11pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_exit.setDefault(True)
        self.bl_exit.setObjectName("bl_exit")
        self.horizontalLayout_2.addWidget(self.bl_exit)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout.setStretch(0, 10)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.label_4 = QtWidgets.QLabel(self.horizontalWidget)
        self.label_4.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_4.setLineWidth(1)
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("avtar.jpg"))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setContentsMargins(8, 9, 9, 9)
        self.formLayout.setHorizontalSpacing(9)
        self.formLayout.setVerticalSpacing(20)
        self.formLayout.setObjectName("formLayout")
        self.bl_ID = QtWidgets.QLabel(self.horizontalWidget)
        self.bl_ID.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_ID.setObjectName("bl_ID")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.bl_ID)
        self.fl_id = QtWidgets.QLineEdit(self.horizontalWidget)
        self.fl_id.setTabletTracking(False)
        self.fl_id.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.fl_id.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);\n"
"")
        self.fl_id.setObjectName("fl_id")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fl_id)
        self.bl_Name = QtWidgets.QLabel(self.horizontalWidget)
        self.bl_Name.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_Name.setObjectName("bl_Name")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.bl_Name)
        self.fl_name = QtWidgets.QLineEdit(self.horizontalWidget)
        self.fl_name.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.fl_name.setObjectName("fl_name")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.fl_name)
        self.bl_Year = QtWidgets.QLabel(self.horizontalWidget)
        self.bl_Year.setAutoFillBackground(False)
        self.bl_Year.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_Year.setObjectName("bl_Year")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.bl_Year)
        self.fl_Year = QtWidgets.QLineEdit(self.horizontalWidget)
        self.fl_Year.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.fl_Year.setObjectName("fl_Year")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.fl_Year)
        self.bl_Major = QtWidgets.QLabel(self.horizontalWidget)
        self.bl_Major.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_Major.setObjectName("bl_Major")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.bl_Major)
        self.fl_major = QtWidgets.QLineEdit(self.horizontalWidget)
        self.fl_major.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.fl_major.setObjectName("fl_major")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.fl_major)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setContentsMargins(8, 10, 8, 9)
        self.formLayout_2.setHorizontalSpacing(11)
        self.formLayout_2.setVerticalSpacing(20)
        self.formLayout_2.setObjectName("formLayout_2")
        self.bl_id = QtWidgets.QLabel(self.horizontalWidget)
        self.bl_id.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_id.setScaledContents(False)
        self.bl_id.setWordWrap(False)
        self.bl_id.setObjectName("bl_id")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.bl_id)
        self.iDLineEdit = QtWidgets.QLineEdit(self.horizontalWidget)
        self.iDLineEdit.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.iDLineEdit.setObjectName("iDLineEdit")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.iDLineEdit)
        self.bl_n = QtWidgets.QLabel(self.horizontalWidget)
        self.bl_n.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_n.setObjectName("bl_n")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.bl_n)
        self.nameLineEdit = QtWidgets.QLineEdit(self.horizontalWidget)
        self.nameLineEdit.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.nameLineEdit.setObjectName("nameLineEdit")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.nameLineEdit)
        self.bl_y = QtWidgets.QLabel(self.horizontalWidget)
        self.bl_y.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_y.setObjectName("bl_y")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.bl_y)
        self.yearLineEdit = QtWidgets.QLineEdit(self.horizontalWidget)
        self.yearLineEdit.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.yearLineEdit.setObjectName("yearLineEdit")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.yearLineEdit)
        self.bl_m = QtWidgets.QLabel(self.horizontalWidget)
        self.bl_m.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_m.setObjectName("bl_m")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.bl_m)
        self.majorLineEdit = QtWidgets.QLineEdit(self.horizontalWidget)
        self.majorLineEdit.setStyleSheet("font: 12pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.majorLineEdit.setObjectName("majorLineEdit")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.majorLineEdit)
        self.verticalLayout_4.addLayout(self.formLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout_4)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.bl_save = QtWidgets.QPushButton(self.horizontalWidget)
        self.bl_save.setStyleSheet("font: 11pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.bl_save.setDefault(True)
        self.bl_save.setObjectName("bl_save")
        self.horizontalLayout_4.addWidget(self.bl_save)
        self.Infor = QtWidgets.QPushButton(self.horizontalWidget)
        self.Infor.setStyleSheet("font: 11pt \"Times New Roman\";\n"
"background-color: rgb(255, 255, 255);")
        self.Infor.setDefault(True)
        self.Infor.setObjectName("Infor")
        self.horizontalLayout_4.addWidget(self.Infor)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 2)
        self.verticalLayout_2.setStretch(3, 1)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout.setStretch(0, 4)
        self.horizontalLayout.setStretch(1, 2)
        #Tạo các sự kiện click tương ứng với các nút nhấn trên GUI
        self.bl_exit.clicked.connect(self.close_and_exit)
        self.bl_cam.clicked.connect(self.load)
        self.Infor.clicked.connect(self.infor)
        self.bl_get.clicked.connect(self.showdata)
        self.bl_save.clicked.connect(self.save)
        self.bl_cut.clicked.connect(self.convert_img)
        self.bl_train.clicked.connect(self.train)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def load(self):
        # Cai dat cac tham so can thiet
        MINSIZE = 100
        THRESHOLD = [0.6, 0.7, 0.7]
        FACTOR = 0.709
        IMAGE_SIZE = 182
        INPUT_IMAGE_SIZE = 160
        CLASSIFIER_PATH = 'C:/Users/MyPC/Downloads/FaceRecogniton/Models/facemodel.pkl'
        FACENET_MODEL_PATH = 'C:/Users/MyPC/Downloads/FaceRecogniton/Models/20180402-114759.pb'

        # Load model da train de nhan dien khuon mat - thuc chat la classifier
        with open(CLASSIFIER_PATH, 'rb') as file:
            model, class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")

        with tf.Graph().as_default():
        # Cai dat GPU neu co
           gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
           sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,log_device_placement=False))

           with sess.as_default():
               # Load model MTCNN phat hien khuon mat
               print('Loading feature extraction model')
               facenet.load_model(FACENET_MODEL_PATH)

               # Lay tensor input va output
               images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
               embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
               phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
               embedding_size = embeddings.get_shape()[1]
               # Cai dat cac mang con
               pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "C:/Users/MyPC/Downloads/FaceRecogniton/src/align")
               people_detected = set()
               person_detected = collections.Counter()

               # Lay hinh anh tu file video
               self.cap = cv2.VideoCapture(0)

               while (self.cap.isOpened()):
                    # Doc tung frame
                    ret, self.frame = self.cap.read() 
                    self.frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    # Phat hien khuon mat, tra ve vi tri trong bounding_boxes
                    bounding_boxes, _ = align.detect_face.detect_face(self.frame, MINSIZE, 
                                                                      pnet, rnet, onet, THRESHOLD, FACTOR)
                    faces_found = bounding_boxes.shape[0]
                    print(faces_found)
                    # Neu co it nhat 1 khuon mat trong frame
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        print(det)
                        print(bb)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            # Cat phan khuon mat tim duoc
                            cropped = self.frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            print(cropped)
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            # Dua vao model de classifier
                            predictions = model.predict_proba(emb_array)
                            self.best_class_indices = np.argmax(predictions, axis=1)
                            self.best_class_probabilities = predictions[np.arange(len(self.best_class_indices)), self.best_class_indices]
                            # Lay ra ten va ty le % cua class co ty le cao nhat
                            self.id = class_names[self.best_class_indices[0]]
                            self.profile = getProfile(self.id)
                            print("Name: {}, Probability: {}".format(self.id, self.best_class_probabilities))  
                            #Lay thoi gian
                            self.now = datetime.now()
                            self.time = self.now.strftime("%d/%m/%Y %H:%M:%S")
                            #Cat va ve khung tren khuon mat
                            self.faces = face_detector.detectMultiScale(self.frame_gray, 1.3, 4) 
                            for (x,y,w,h) in self.faces:
                                cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0, 255, 0), 2) 
                                self.convert = self.frame[y+2: y+h-2, x+2: x+w-2]  
                                if self.best_class_probabilities > 0.5:
                                    self.input_infor()
                                    self.cutFace()
                                    self.sql_data()
                                else:
                                    self.unknown()
                                self.update()
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            
    #Xuất thông tin người dùng nhận diện thành công   
    def input_infor(self):
        self.fl_id.setText(str(self.profile[0]))
        self.fl_name.setText(str(self.profile[1]))
        self.fl_Year.setText(str(self.profile[2]))
        self.fl_major.setText(str(self.profile[3]))

    #Xuất thông tin người dùng không có thông tin trong tập dữ liệu    
    def unknown(self):
        text0 = 'Khong Co Thong Tin'
        ps.putBText(self.frame, text0, text_offset_x=10, text_offset_y=140, 
                    vspace=20, hspace=10,font_scale=0.7,background_RGB=(10,20,222),text_RGB=(255,255,255))
        self.fl_id.setText('')
        self.fl_name.setText('')
        self.fl_Year.setText('')
        self.fl_major.setText('')
        img_path= 'C:/Users/MyPC/Downloads/FaceRecogniton/src/avtar.jpg'
        img0 = QtGui.QImage(img_path)  
        self.label_4.setPixmap(QtGui.QPixmap.fromImage(img0))
    
    #Hiển thị khuôn mặt người dùng nhận diện thành công
    def cutFace(self):
        image = cv2.cvtColor(self.convert, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(153,154))
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_RGB888)
        self.label_4.setPixmap(QtGui.QPixmap.fromImage(img))
    
    #Tạo cột Time để ghi nhận thông tin khi người dùng nhận diện thành công
    def sql_data(self):
        self.time1 = self.now.strftime("%d/%m/%Y")
        #Kết nối tới cơ sở dữ liệu
        conn = sqlite3.connect('C:/Users/MyPC/Downloads/FaceRecogniton/src/DataSet.db')
        cursor = conn.cursor()
        #Kiểm tra đã có cột time chưa
        cursor.execute("PRAGMA table_info(DataSet)")
        columns = cursor.fetchall()
        time_column_exists = any(column[1] == str('Time_' + self.time1) for column in columns)
        #Nếu chưa có tạo cột time mới
        if not time_column_exists:
            cursor.execute("ALTER TABLE DataSet ADD COLUMN 'Time_" + self.time1 + "' TEXT")
        #so sanh
        cursor.execute("SELECT id FROM DataSet WHERE id = ?", (self.id,))
        result = cursor.fetchone()
        if result is not None:
            cursor.execute("SELECT `Time_" + self.time1 + "` FROM DataSet WHERE id = ?", (self.id,))
            result = cursor.fetchone()
            if result[0] is not None:
                text5 = 'Du Lieu Da Duoc Nhap'
                ps.putBText(self.frame, text5, text_offset_x=10, text_offset_y=220, 
                            vspace=20, hspace=10,font_scale=0.7,background_RGB=(228,20,222),text_RGB=(255,255,255))
            else:
                cursor.execute("UPDATE DataSet SET 'Time_" + self.time1 + "' = ? WHERE id = ?", (self.time, self.id))
                conn.commit()
                text4 = 'Dang Nhap Du Lieu'
                ps.putBText(self.frame, text4, text_offset_x=10, text_offset_y=180, 
                            vspace=20, hspace=10,font_scale=0.7,background_RGB=(228,20,222),text_RGB=(255,255,255))
        else:
            print("Id người dùng không tồn tại!")
        conn.commit()
        conn.close()

    #Hiện tỷ lệ nhận diện và thời gian thực lên màn hình nhận diện
    def update(self):
        text = 'Ty Le Nhan Dien: ' + str(self.best_class_probabilities)
        ps.putBText(self.frame, text, text_offset_x=10, text_offset_y=25, vspace=20, 
                    hspace=10,font_scale=0.7,background_RGB=(10,20,222),text_RGB=(255,255,255))
        text1 = 'Thoi Gian Thuc: ' + str(self.time)
        ps.putBText(self.frame, text1, text_offset_x=10, text_offset_y=80, vspace=20, 
                    hspace=10,font_scale=0.7,background_RGB=(228,20,222),text_RGB=(255,255,255))
        self.setPhoto(self.frame)
    
    #Cài đặt kích cỡ màn hình nhận diện
    def setPhoto(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(653,514))
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_RGB888)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(img))

    #Tắt GUI
    def close_and_exit(self):
        exit(0)    

   #Hiển thị bảng lưu thông tin người dùng
    def infor(self):
        self.sqlite3_exe_path = "C:/Program Files/DB Browser for SQLite/DB Browser for SQLite.exe"
        self.db_file_path = "C:/Users/MyPC/Downloads/FaceRecogniton/src/DataSet.db"
        subprocess.call([self.sqlite3_exe_path, self.db_file_path])
   
   #Hiển thị tập dữ liệu
    def showdata(self):
        self.folder_path = "C:/Users/MyPC/Downloads/FaceRecogniton/src/Dataset"
        os.startfile(self.folder_path)
   
   #Huấn luyện tập dữ liệu
    def train(self):
        self.command = "python C:/Users/MyPC/Downloads/FaceRecogniton/src/classifier.py TRAIN C:/Users/MyPC/Downloads/FaceRecogniton/src/Dataset/FaceData/processed C:/Users/MyPC/Downloads/FaceRecogniton/Models/20180402-114759.pb C:/Users/MyPC/Downloads/FaceRecogniton/Models/facemodel.pkl --batch_size 1000"
        subprocess.run(self.command, shell=True)
        img_path0= 'C:/Users/MyPC/Downloads/FaceRecogniton/src/train.png'
        img01 = QtGui.QImage(img_path0)  
        self.label_4.setPixmap(QtGui.QPixmap.fromImage(img01))
   
   #Lưu thông tin người dùng mới
    def save(self):
        self.ID = self.iDLineEdit.text()
        self.Name = self.nameLineEdit.text()
        self.Year = self.yearLineEdit.text()
        self.Major = self.majorLineEdit.text()
        conn = sqlite3.connect('C:/Users/MyPC/Downloads/FaceRecogniton/src/DataSet.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO DataSet (ID, Name, Year, Major) VALUES (?, ?, ?, ?)", 
                       (self.ID, self.Name, self.Year, self.Major))
        conn.commit()
        cursor.close()
        conn.close()
        self.iDLineEdit.clear()
        self.nameLineEdit.clear()
        self.yearLineEdit.clear()
        self.majorLineEdit.clear()

    #Chỉnh sửa ảnh người dùng mới
    def convert_img(self):
        self.convert = "python C:/Users/MyPC/Downloads/FaceRecogniton/src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25"
        subprocess.run(self.convert, shell=True)
        img_path1= 'C:/Users/MyPC/Downloads/FaceRecogniton/src/convert.png'
        img02 = QtGui.QImage(img_path1)  
        self.label_4.setPixmap(QtGui.QPixmap.fromImage(img02))
                               
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Face Recogniton in Real-time"))
        self.bl_get.setText(_translate("Dialog", "Show data"))
        self.bl_cut.setText(_translate("Dialog", "Convert"))
        self.bl_train.setText(_translate("Dialog", "Train data"))
        self.bl_cam.setText(_translate("Dialog", "Face Recogniton"))
        self.bl_exit.setText(_translate("Dialog", "Exit"))
        self.bl_ID.setText(_translate("Dialog", "ID:"))
        self.bl_Name.setText(_translate("Dialog", "Name:"))
        self.bl_Year.setText(_translate("Dialog", "Year:"))
        self.bl_Major.setText(_translate("Dialog", "Major:"))
        self.bl_id.setText(_translate("Dialog", "Input ID:"))
        self.bl_n.setText(_translate("Dialog", "Input Name:"))
        self.bl_y.setText(_translate("Dialog", "Input Year:"))
        self.bl_m.setText(_translate("Dialog", "Input Major:"))
        self.bl_save.setText(_translate("Dialog", "Save"))
        self.Infor.setText(_translate("Dialog", "Infor"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
