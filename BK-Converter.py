#!/usr/bin/env python

# -*- coding: utf-8 -*-

from PIL import Image, ImageQt, ImageDraw
import os, sys, traceback, regex, subprocess, PyPDF2
import numpy as np
from natsort import natsorted
import configparser

from PyQt5.QtWidgets import (QWidget, QLabel, QFileDialog, QPushButton, QHBoxLayout, QApplication, QFrame,
                             QLineEdit, QSlider, QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
                             QSpacerItem, QSizePolicy, QSpinBox, QComboBox, QCheckBox, QGroupBox, QProgressBar,
                             QGridLayout, QAction, QMainWindow, QScrollBar, QInputDialog, QMessageBox)
from PyQt5.QtCore import (Qt, QObject, pyqtSignal, QRunnable, pyqtSlot, QThreadPool)
from PyQt5.QtGui import (QBrush, QColor)



# If autoopen is set to some directory, that directory will be loaded on startup. Only used for debugging
autoopen = None
#autoopen = ""


def parse_image_types(text):
    text = regex.sub(r"[^\w;]", "", text)
    text = text.split(";")
    return tuple(x.lower() for x in text if x != "")

# Set program variables
class VAR:

    settings_file = os.path.join(os.getcwd(), 'BK_settings.ini')

    settings = configparser.ConfigParser()
    if os.path.isfile(settings_file):
        settings.read(settings_file)
    else:
        settings['DEFAULT'] = {
            'Threshold': '140',
            'Page Offset': '0',
            'Processing Boxsize': '60',
            'Image Sensitivity': '90',
            'Image Rotation': '0',
            'Text cropping-Left': '0',
            'Text cropping-Top': '0',
            'Text cropping-Right': '0',
            'Text cropping-Bottom': '0',
            'Allowable Image Types': 'png; jpg',
            'Working Directory': os.path.join(os.path.expanduser("~"), "Desktop"),
            'Path to Tesseract': 'tesseract',
            'Path to Ghostscript': 'gswin64c',
            'Last Directory': os.path.join(os.path.expanduser("~"), "Desktop")
        }
        with open(settings_file, 'w') as f:
            settings.write(f)

    setd = settings['DEFAULT']

    thres = setd.getint('Threshold')
    offset = setd.getint('Page Offset')
    boxsize = setd.getint('Processing Boxsize')
    im_sens = setd.getint('Image Sensitivity')
    rotation = setd.getint('Image Rotation')
    cropbox = (setd.getint('Text cropping-Left'),
               setd.getint('Text cropping-Top'),
               setd.getint('Text cropping-Right'),
               setd.getint('Text cropping-Bottom'))
    im_types = parse_image_types(setd['Allowable Image Types'])
    filename = ''
    final_directory = ''
    temp_dir = ''
    dirname_processed = 'BKprocessed'
    dirname_cropped = 'BKcropped'
    dirname_ocr = 'BKocr'
    temp_base_dir = setd['Working Directory']
    path_to_tess = setd['Path to Tesseract']
    path_to_GS = setd['Path to Ghostscript']


def load_images(progress_callback):
    """
    Load images into memory via Pillow
    """

    images = []

    if autoopen:
        os.chdir(autoopen)
        filenames = os.listdir()[:5]
    else:
        filenames = [name for name in os.listdir(".") if name.lower().endswith(VAR.im_types)]

    filenames = natsorted(filenames)

    total = len(filenames)

    # this counter is for the progress bar
    i = 0

    for file in filenames:
        i += 1
        im = Image.open(file).convert("L")
        images.append(im)
        progress_callback.emit(int((i / total) * 100))

    return images


def load_pdf(input_fn, output_fn, progress_callback):
    """
    Calls Ghostscript on a PDF in order to turn it into a series of images, which are then loaded via load_images()
    """

    GSargs = [VAR.path_to_GS,
            "-q",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pnggray",
            "-r300",
            "-sOutputFile=" + output_fn,
            input_fn]

    subprocess.call(GSargs)

    os.chdir(VAR.temp_dir)
    images = load_images(progress_callback)

    return images


def split_image(im, pointer=None):
    """
    If the split image box is checked, this function divides the image into two pages with the given offset
    """

    # rotate the image for any non-zero value entered in the rotation box
    if VAR.rotation:
        im = im.rotate(VAR.rotation, expand=True)

    # put the page split point at the global offset value unless there is a local value in local_adj
    if not window.this_page_only_check.isChecked():
        LR_split_point = int(im.width / 2) + VAR.offset
    else:
        LR_split_point = int(im.width / 2) + window.local_adj[pointer]["offset"]

    # here we make sure that each page has some minimum width, which is 4 pixels
    if LR_split_point > im.width:
        LR_split_point = im.width - 4
    elif LR_split_point < 4:
        LR_split_point = 4

    # setup the boxes for cropping
    boxL = (0, 0, LR_split_point, im.height)
    boxR = (LR_split_point, 0, im.width, im.height)

    # do the actual cropping
    outL = im.crop(box=boxL)
    outR = im.crop(box=boxR)

    return [outL, outR]


def scale_pil(im, height):
    """
    Scales the image given a certain height
    """
    # clamp the height to 10 pixels minimum
    if height < 10:
        height = 10

    # compute the ratio and calculate the new width x
    ratio = im.width / im.height
    x = int(height * ratio)

    # resize and return
    return im.resize((int(x), int(height)))


def check_for_image_func(ia, xmax, ymax, boxsize):
    """
    called within process_image(), calculates the mean value of a box and if that value is below im_sens, then
    it is considered part of an image. The idea is that only images will have large dark areas, whereas areas of
    just text will be mostly background (and mostly brighter)

    This function returns a rectangle of coordinates identified as part of an image
    """

    coord = []
    for x in range(0, xmax, boxsize):
        for y in range(0, ymax, boxsize):
            # here we make sure that no box of pixels within 160 pixels of the left or right edge
            # will be picked up. Dark areas close to the edge are likely to be scanning artifacts
            # rather than images
            if 160 < y < (ymax - 160):
                arr = ia[x:x + boxsize, y:y + boxsize]
                mean = arr.mean()
                # if the mean is darker than VAR.im_sens, we add its box to an array of coordinates
                if mean < VAR.im_sens:
                    coord.append((x, y))

    # if any coordinates have been added to this array, we find the min and max values of the coordinates
    # in order to get a box. Pixels within this box will not be processed.
    if len(coord) != 0:
        picx = np.array([x for x, y in coord])
        picy = np.array([y for x, y in coord])
        return picx.min() - boxsize, picy.min() - boxsize, picx.max() + boxsize, picy.max() + boxsize, True
    else:
        return xmax, ymax, xmax, ymax, False


def process_dropped(im):
    im = im.point(lambda x: 0 if x < VAR.thres else 100, 'L')
    return im

def process_image(algorithm, check_for_images, im, ratio=None, final=False):
    """
    processes the image if the process image box is checked
    """

    # if ratio is passed, this is a quick preview and will be resized later. Therefore, the boxsize needs
    # to be adjusted accordingly
    if ratio:
        boxsize = int(VAR.boxsize * ratio)
    else:
        boxsize = VAR.boxsize

    if window.processGB.isChecked():
        # simple threshold algorithm
        if algorithm == 0:

            if check_for_images:

                ia = np.asarray(im).copy()
                xmax, ymax = ia.shape

                rect_xmin, rect_ymin, rect_xmax, rect_ymax, hasImage = check_for_image_func(ia, xmax, ymax, boxsize)

                # for every box of size boxsize in the image...
                for x in range(0, xmax, boxsize):
                    for y in range(0, ymax, boxsize):
                        # if we aren't within an image
                        if not (rect_xmin < x < rect_xmax and rect_ymin < y < rect_ymax):
                            # then convert every pixel above the threshold to 255 and below to 0
                            arr = ia[x:x + boxsize, y:y + boxsize]
                            arr[arr < VAR.thres] = 0
                            arr[arr >= VAR.thres] = 255

                im = Image.fromarray(ia)
            else:
                # if there are no images to look for, it is not necessary to process the images in boxes,
                # we can just do it all at once without numpy
                im = im.point(lambda x: 0 if x < VAR.thres else 255, '1')
                hasImage = False

        # mean of box algorithm
        elif algorithm == 1:

            ia = np.asarray(im).copy()
            xmax, ymax = ia.shape

            if check_for_images:
                rect_xmin, rect_ymin, rect_xmax, rect_ymax, hasImage = check_for_image_func(ia, xmax, ymax, boxsize)
            else:
                rect_xmin, rect_ymin, rect_xmax, rect_ymax, hasImage = xmax, ymax, xmax, ymax, False

            # same as above
            for x in range(0, xmax, boxsize):
                for y in range(0, ymax, boxsize):
                    # if we aren't within an image
                    if not (rect_xmin < x < rect_xmax and rect_ymin < y < rect_ymax):
                        # find the mean first and then add the threshold to that to get the final threshold,
                        # which will be different for every box
                        arr = ia[x:x + boxsize, y:y + boxsize]
                        mean = arr.mean()
                        arr[arr < mean + VAR.thres] = 0
                        arr[arr >= mean + VAR.thres] = 255

            im = Image.fromarray(ia)

    # we add the red crop lines if the copping box is checked, but not if this is a quick preview and
    # definitely not if this is the final output
    if window.cropGB.isChecked() and not final:
        im = add_croplines(im, ratio)

    # for final output, image will be grayscale if it has an image, otherwise just black and white
    if final:
        if hasImage:
            im = im.convert("L")
        else:
            im = im.convert("1")

    return im


def add_croplines(im, ratio):
    """
    add red crops to the preview image in the GUI
    """

    if window.this_page_only_check.isChecked():
        box = window.local_adj[window.pointer]["cropbox"]
        boxL = box[0]
        boxT = box[1]
        boxR = box[2]
        boxB = box[3]
    else:
        boxL = VAR.cropbox[0]
        boxT = VAR.cropbox[1]
        boxR = VAR.cropbox[2]
        boxB = VAR.cropbox[3]

    im = im.convert("RGB")
    draw = ImageDraw.Draw(im)

    fill = (255, 0, 0)
    line_width = 6

    if ratio:
        boxL = int(boxL * ratio)
        boxT = int(boxT * ratio)
        boxR = int(boxR * ratio)
        boxB = int(boxB * ratio)
        line_width = int(line_width * ratio)

    # Draw the lines, and clamp values if they exceed image dimensions
    if boxL:
        if boxL < im.width:
            draw.line((boxL, 0, boxL, im.height), fill=fill, width=line_width)
        else:
            draw.line((im.width, 0, im.width, im.height), fill=fill, width=line_width)
    if boxR:
        if boxR < im.width:
            draw.line((im.width-boxR, 0, im.width-boxR, im.height), fill=fill, width=line_width)
        else:
            draw.line((0, 0, 0, im.height), fill=fill, width=line_width)
    if boxT:
        if boxT < im.height:
            draw.line((0, boxT, im.width, boxT), fill=fill, width=line_width)
        else:
            draw.line((0, im.height, im.width, im.height), fill=fill, width=line_width)
    if boxB:
        if boxB < im.height:
            draw.line((0, im.height-boxB, im.width, im.height-boxB), fill=fill, width=line_width)
        else:
            draw.line((0, 0, im.width, 0), fill=fill, width=line_width)

    del draw

    return im


def final_crop(im, pointer=None):
    """
    for final processing, we actually crop the image instead of adding crop lines, if any box has a non-zero value
    """

    if window.this_page_only_check.isChecked():
        box = window.local_adj[window.pointer]["cropbox"]
        boxL = box[0]
        boxT = box[1]
        boxR = box[2]
        boxB = box[3]
    else:
        boxL = VAR.cropbox[0]
        boxT = VAR.cropbox[1]
        boxR = VAR.cropbox[2]
        boxB = VAR.cropbox[3]


    if boxL >= im.width:
        boxL = im.width - 1
    if boxR >= im.width:
        boxR = im.width - 1
    if boxT >= im.height:
        boxT = im.height - 1
    if boxB >= im.height:
        boxB = im.height - 1

    if boxL + boxR >= im.width:
        boxR = im.width - boxL - 1
    if boxT + boxB >= im.height:
        boxB = im.height - boxT - 1

    if boxL or boxR or boxT or boxB:
        rect = (boxL, boxT, im.width - boxR, im.height - boxB)
        im = im.crop(box=rect)

    return im


class Converter(QMainWindow):
    def __init__(self, parent=None):
        super(Converter, self).__init__(parent)

        MainWidget = QWidget()
        self.setCentralWidget(MainWidget)

        self.resize(1200, 800)
        frameStyle = QFrame.Sunken | QFrame.Panel

        # The top row of buttons for opening files/directories
        self.open_pdfbtn = QPushButton("Open PDF")
        self.directory_btn = QPushButton("Open Image Directory")
        self.directory_btn.setMinimumWidth(160)
        self.dirlabel = QLabel()
        self.dirlabel.setText("")
        self.dirlabel.setFrameStyle(frameStyle)
        self.dirlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # The name row with miscellaneous functions
        row1 = QHBoxLayout()
        self.name_label = QLabel()
        self.name_label.setText("Name:")
        self.name_input = QLineEdit()
        self.name_input.setMinimumWidth(80)
        self.name_input.setContentsMargins(0, 0, 10, 0)
        self.name_input.setText("MyBook")
        self.name_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.page_input_label = QLabel()
        self.page_input_label.setText("Page Offset:")
        self.page_input = QSpinBox()
        self.page_input.setMinimum(-999)
        self.page_input.setMaximum(999)
        self.page_input.setValue(0)
        self.rotate_label = QLabel()
        self.rotate_label.setText("Rotate:")
        self.rotate_box = QSpinBox()
        self.rotate_box.setMinimum(-360)
        self.rotate_box.setMaximum(360)
        self.rotate_box.setValue(0)
        self.rotate_box.setSingleStep(10)
        row1.addWidget(self.name_label)
        row1.addWidget(self.name_input)
        row1.addWidget(self.page_input_label)
        row1.addWidget(self.page_input)
        row1.addWidget(self.rotate_label)
        row1.addWidget(self.rotate_box)
        #row1.setContentsMargins(0, 0, 0, 10)
        #row1.addStretch()

        global_row = QVBoxLayout()
        row2 = QHBoxLayout()
        self.this_page_only_check = QCheckBox()
        self.this_page_only_check.setText("Apply parameters to this page only?")
        self.this_page_only_check.setChecked(False)
        self.this_page_only_reset = QPushButton("Reset all")
        self.drop_pageL_btn = QPushButton("Drop Left")
        self.drop_pageR_btn = QPushButton("Drop Right")
        self.drop_pageL_btn.setCheckable(True)
        self.drop_pageR_btn.setCheckable(True)
        row2.addWidget(self.this_page_only_check)
        row2.addWidget(self.this_page_only_reset)
        row2.addStretch()
        row2.addWidget(self.drop_pageL_btn)
        row2.addWidget(self.drop_pageR_btn)
        row2.addStretch()
        global_row.addLayout(row1)
        global_row.addLayout(row2)

        # the split page box
        self.splitpageGB = QGroupBox("Split Pages?")
        self.splitpageGB.setCheckable(True)
        splitpageGBvlayout = QVBoxLayout()
        self.offset_slider = MySlider(Qt.Horizontal)
        self.offset_slider.setSingleStep(1)
        self.offset_min, self.offset_max, self.offset_slider.initval = -999, 999, 0
        self.offset_slider.setMinimum(self.offset_min)
        self.offset_slider.setMaximum(self.offset_max)
        self.offset_slider.setTickPosition(QSlider.TicksBothSides)
        self.offset_slider.setTickInterval(80)
        self.offset_slider.setFocusPolicy(Qt.StrongFocus)
        self.offset_slider.setValue(self.offset_slider.initval)
        self.offset_slider.setMinimumWidth(100)
        splitpageGBvlayout.addWidget(self.offset_slider)
        splitpageGBlabellayout = QHBoxLayout()
        self.offset_label = QLabel()
        self.offset_label.setText("Offset:")
        self.offset_spinbox = QSpinBox()
        self.offset_spinbox.setMinimum(self.offset_min)
        self.offset_spinbox.setMaximum(self.offset_max)
        self.offset_spinbox.setValue(self.offset_slider.initval)
        splitpageGBlabellayout.addStretch()
        splitpageGBlabellayout.addWidget(self.offset_label)
        splitpageGBlabellayout.addWidget(self.offset_spinbox)
        splitpageGBlabellayout.addStretch()
        splitpageGBvlayout.addLayout(splitpageGBlabellayout)
        self.splitpageGB.setLayout(splitpageGBvlayout)

        # the box for cropping
        self.cropGB = QGroupBox("Crop for OCR?")
        self.cropGB.setCheckable(True)
        self.cropGB.setChecked(False)
        cropgrid = QGridLayout()
        self.cropLbox = QSpinBox()
        self.cropRbox = QSpinBox()
        self.cropTbox = QSpinBox()
        self.cropBbox = QSpinBox()
        self.cropboxes = [self.cropLbox, self.cropRbox, self.cropTbox, self.cropBbox]
        for box in self.cropboxes:
            box.setMinimum(0)
            box.setMaximum(999)
            box.setValue(0)
            box.setSingleStep(10)
            #box.setMaximumWidth(60)
            box.valueChanged.connect(self.cropbox_changed)
        self.cropGB.setStyleSheet("QSpinBox {border:1px solid #cccccc}")
        cropgrid.addWidget(self.cropTbox, 0, 1)
        cropgrid.addWidget(self.cropLbox, 1, 0)
        cropgrid.addWidget(self.cropRbox, 1, 2)
        cropgrid.addWidget(self.cropBbox, 2, 1)
        self.cropGB.setLayout(cropgrid)

        # the processing box
        self.processGB = QGroupBox("Process?")
        self.processGB.setCheckable(True)
        #self.processGB.setMaximumHeight(self.cropGB.height() + self.name_input.height())
        self.processGB.setMaximumHeight(200)
        process_mainHlayout = QHBoxLayout()
        process_firstcol = QVBoxLayout()
        self.algorithm_label = QLabel()
        self.algorithm_label.setText("Algorithm:")
        self.algorithm = QComboBox()
        self.algorithm.addItems(["Simple threshold", "Mean of box"])
        self.algorithm.setMaximumWidth(240)
        self.algorithm_imagebox = QCheckBox()
        self.algorithm_imagebox.setText("Look for images?")
        process_firstcol_imsens = QHBoxLayout()
        self.algorithm_imsens_label = QLabel()
        self.algorithm_imsens_label.setText("Sensitivity:")
        self.algorithm_imsens = MySlider(Qt.Horizontal)
        self.algorithm_imsens.setMinimum(70)
        self.algorithm_imsens.setMaximum(110)
        self.algorithm_imsens.initval = 90
        self.algorithm_imsens.setValue(self.algorithm_imsens.initval)
        self.algorithm_imsens.setMaximumWidth(200)
        self.algorithm_imsens_val = QLabel()
        self.algorithm_imsens_val.setText("{0:03d}".format(self.algorithm_imsens.value()))
        process_firstcol_imsens.addWidget(self.algorithm_imsens_label)
        process_firstcol_imsens.addWidget(self.algorithm_imsens)
        process_firstcol_imsens.addWidget(self.algorithm_imsens_val)
        process_firstcol_imsens.addStretch()
        process_firstcol.addWidget(self.algorithm_label)
        process_firstcol.addWidget(self.algorithm)
        process_firstcol.addWidget(self.algorithm_imagebox)
        process_firstcol.addLayout(process_firstcol_imsens)
        process_mainHlayout.addLayout(process_firstcol)

        thres_vbox = QVBoxLayout()
        self.thres_label = QLabel()
        self.thres_label.setText("Threshold:")
        self.thres_slider = MySlider(Qt.Vertical)
        self.thres_slider.setSingleStep(1)
        self.thres_min, self.thres_max, self.thres_slider.initval = 1, 255, 140
        self.thres_slider.alg0val, self.thres_slider.alg1val = 140, -20
        self.thres_slider.setMinimum(self.thres_min)
        self.thres_slider.setMaximum(self.thres_max)
        self.thres_slider.setValue(self.thres_slider.initval)
        self.thres_slider.setMaximumHeight(80)
        self.thres_slider.setFocusPolicy(Qt.StrongFocus)
        self.thres_spinbox = QSpinBox()
        self.thres_spinbox.setMinimum(self.thres_min)
        self.thres_spinbox.setMaximum(self.thres_max)
        self.thres_spinbox.setValue(self.thres_slider.initval)
        thres_vbox.addWidget(self.thres_slider)
        thres_vbox.addWidget(self.thres_spinbox)

        boxsize_vbox = QVBoxLayout()
        self.boxsize_label = QLabel()
        self.boxsize_label.setText("Box size:")
        self.boxsize_slider = MySlider(Qt.Vertical)
        self.boxsize_slider.setSingleStep(1)
        self.boxsize_min, self.boxsize_max, self.boxsize_slider.initval = 25, 150, 60
        self.boxsize_slider.setMinimum(self.boxsize_min)
        self.boxsize_slider.setMaximum(self.boxsize_max)
        self.boxsize_slider.setValue(self.boxsize_slider.initval)
        self.boxsize_slider.setMaximumHeight(80)
        self.boxsize_spinbox = QSpinBox()
        self.boxsize_spinbox.setMinimum(self.boxsize_min)
        self.boxsize_spinbox.setMaximum(self.boxsize_max)
        self.boxsize_spinbox.setValue(self.boxsize_slider.initval)
        boxsize_vbox.addWidget(self.boxsize_slider)
        boxsize_vbox.addWidget(self.boxsize_spinbox)

        process_mainHlayout.addWidget(self.thres_label)
        process_mainHlayout.addLayout(thres_vbox)
        process_mainHlayout.addWidget(self.boxsize_label)
        process_mainHlayout.addLayout(boxsize_vbox)
        self.processGB.setLayout(process_mainHlayout)

        # the row with page numbers
        self.lpage_label = QLabel()
        self.rpage_label = QLabel()
        self.rpage_label.setContentsMargins(60, 0, 0, 0)

        # the main viewer
        self.scene = QGraphicsScene()
        self.view = MyGraphicsView(self.scene)
        self.view.setInteractive(True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graybg = QBrush(QColor(Qt.lightGray))
        self.view.setBackgroundBrush(self.graybg)
        self.progress_bar = QProgressBar()

        # page scroll bar
        self.viewer_scrollbar = QScrollBar()
        self.viewer_scrollbar.setOrientation(Qt.Horizontal)

        # the previous and next buttons
        self.prev_img_btn = QPushButton("Prev")
        self.gotopage_labelL = QLabel()
        self.gotopage_labelL.setText("Go to page")
        self.gotopage_labelL.setMaximumWidth(66)
        self.gotopage_box = QSpinBox()
        self.gotopage_box.setMaximumWidth(50)
        self.gotopage_labelR = QLabel("of {}".format("0"))
        self.gotopage_labelR.setMaximumWidth(40)
        self.next_img_btn = QPushButton("Next")

        # the final row of processing and OCR buttons
        self.process_btn = QPushButton("Process Images")
        self.ocr_btn = QPushButton("OCR Images")
        self.ocr_pdf_box = QCheckBox()
        self.ocr_pdf_box.setText("Pdf?")
        self.ocr_pdf_box.setChecked(True)
        self.ocr_txt_box = QCheckBox()
        self.ocr_txt_box.setText("Text?")
        self.ocr_txt_box.setChecked(True)
        self.processing_label = QLabel()
        self.process_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        # disable widgets on startup when nothing is loaded
        self.splitpageGB.setEnabled(False)
        self.cropGB.setEnabled(False)
        self.processGB.setEnabled(False)
        self.bottom_controls = [self.prev_img_btn, self.next_img_btn, self.process_btn, self.ocr_btn, self.ocr_pdf_box,
                                self.ocr_txt_box, self.gotopage_box, self.viewer_scrollbar]
        for control in self.bottom_controls:
            control.setEnabled(False)
        self.name_controls = [self.name_label, self.name_input, self.page_input_label, self.page_input, self.rotate_label, self.rotate_box]
        for control in self.name_controls:
            control.setEnabled(False)

        # set connects
        self.page_input.valueChanged.connect(self.update_page_offset)
        self.name_input.textChanged.connect(self.update_page_offset)
        self.this_page_only_reset.clicked.connect(self.this_page_only_reset_func)
        self.drop_pageL_btn.clicked.connect(self.drop_pageL)
        self.drop_pageR_btn.clicked.connect(self.drop_pageR)
        self.rotate_box.valueChanged.connect(self.rotate)
        self.open_pdfbtn.clicked.connect(self.openpdf)
        self.directory_btn.clicked.connect(self.setExistingDirectory)
        self.this_page_only_check.clicked.connect(self.this_page_only_func)
        self.algorithm.currentIndexChanged.connect(self.algorithm_changed)
        self.algorithm_imagebox.stateChanged.connect(self.imagebox_func)
        self.algorithm_imsens.valueChanged.connect(self.imsens)
        self.thres_slider.sliderMoved.connect(self.setThreshold)
        self.thres_slider.sliderReleased.connect(self.view_current)
        self.thres_spinbox.valueChanged.connect(self.setThreshold_spinbox)
        self.offset_slider.valueChanged.connect(self.setOffset)
        self.offset_spinbox.valueChanged.connect(self.offset_slider.setValue)
        self.boxsize_slider.sliderMoved.connect(self.setboxsize)
        self.boxsize_slider.sliderReleased.connect(self.view_current)
        self.boxsize_spinbox.valueChanged.connect(self.setboxsize_spinbox)
        self.splitpageGB.clicked.connect(self.split_page_toggle)
        self.cropGB.clicked.connect(self.view_current)
        self.processGB.clicked.connect(self.view_current)
        self.wheelEvent = self.wheel_event
        self.viewer_scrollbar.valueChanged.connect(self.viewer_slider_moved)
        self.gotopage_box.valueChanged.connect(self.viewer_box_changed)
        self.viewer_scrollbar.sliderReleased.connect(self.viewer_slider_released)
        self.prev_img_btn.clicked.connect(self.prev)
        self.next_img_btn.clicked.connect(self.next)
        self.process_btn.clicked.connect(self.process)
        self.ocr_btn.clicked.connect(self.ocr)

        # set layouts
        row_directory = QHBoxLayout()
        row_directory.addWidget(self.open_pdfbtn)
        row_directory.addWidget(self.directory_btn)
        self.dirlabel.setHidden(True)
        row_directory.addWidget(self.dirlabel)
        row_directory.addWidget(self.progress_bar)

        row_split_and_crop = QHBoxLayout()
        row_split_and_crop.addWidget(self.splitpageGB)
        row_split_and_crop.addWidget(self.cropGB)

        global_row.addLayout(row_split_and_crop)

        entire_control_row = QHBoxLayout()
        entire_control_row.addLayout(global_row)
        entire_control_row.addWidget(self.processGB)
        entire_control_row.setContentsMargins(0, 10, 0, 10)

        self.row_viewer_labels = QHBoxLayout()
        self.row_viewer_labels.addWidget(self.lpage_label)
        self.row_viewer_labels.addWidget(self.rpage_label)
        self.row_viewer_labels.setAlignment(Qt.AlignHCenter)

        row_viewer = QHBoxLayout()
        row_viewer.addWidget(self.view)

        row_viewer_buttons = QHBoxLayout()
        row_viewer_buttons.addWidget(self.prev_img_btn)
        row_viewer_buttons.addWidget(self.gotopage_labelL)
        row_viewer_buttons.addWidget(self.gotopage_box)
        row_viewer_buttons.addWidget(self.gotopage_labelR)
        row_viewer_buttons.addWidget(self.next_img_btn)

        row_process = QHBoxLayout()
        row_process.addWidget(self.process_btn)
        row_process.addWidget(self.ocr_btn)
        row_process.addWidget(self.ocr_pdf_box)
        row_process.addWidget(self.ocr_txt_box)

        row_process.addWidget(self.processing_label)
        row_process.addStretch()

        main_vlayout = QVBoxLayout()
        main_vlayout.addLayout(row_directory)
        main_vlayout.addLayout(entire_control_row)
        main_vlayout.addLayout(self.row_viewer_labels)
        main_vlayout.addLayout(row_viewer)
        main_vlayout.addWidget(self.viewer_scrollbar)
        main_vlayout.addLayout(row_viewer_buttons)
        main_vlayout.addLayout(row_process)

        MainWidget.setLayout(main_vlayout)

        self.setWindowTitle("BK Converter")

        # create menu and actions
        self.createActions()
        self.createMenu()

        self.threadpool = QThreadPool()

        if autoopen:
            self.set_view()

    def setWorkingDir(self):
        options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(self, "Set Working Directory",
                                                     VAR.temp_base_dir, options=options)
        if directory:
            VAR.temp_base_dir = directory
            VAR.setd['Working Directory'] = directory
            with open(VAR.settings_file, 'w') as f:
                VAR.settings.write(f)

    def setTessDir(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Set Path to Tesseract (for OCR)", VAR.path_to_tess,
                                                  "All Files (*)", options=options)
        if fileName:
            VAR.path_to_tess = fileName
            VAR.setd['Path to Tesseract'] = fileName
            with open(VAR.settings_file, 'w') as f:
                VAR.settings.write(f)

    def setGSDir(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Set Path to Ghostscript (for PDF input)", VAR.path_to_GS,
                                                  "All Files (*)", options=options)
        if fileName:
            VAR.path_to_GS = fileName
            VAR.setd['Path to Ghostscript'] = fileName
            with open(VAR.settings_file, 'w') as f:
                VAR.settings.write(f)

    def setImageTypes(self):
        imtypes = "; ".join(map(str, VAR.im_types))

        text, ok = QInputDialog.getText(self, "Acceptable Image Types",
                                        "Enter image types separated by a semi-colon:", QLineEdit.Normal, imtypes)

        if ok and text != '':
            text = parse_image_types(text)
            VAR.im_types = text
            VAR.setd['Allowable Image Types'] = "; ".join(map(str, VAR.im_types))
            with open(VAR.settings_file, 'w') as f:
                VAR.settings.write(f)

    def createActions(self):
        self.openFolderAct = QAction("&Open Folder", self, triggered=self.setExistingDirectory)

        self.setWorkingDirAct = QAction("Set &Working Directory", self, triggered=self.setWorkingDir)
        self.setTessDirAct = QAction("Set Path to &Tesseract", self, triggered=self.setTessDir)
        self.setGSDirAct = QAction("Set Path to &Ghostscript", self, triggered=self.setGSDir)
        self.setImageTypesAct = QAction("Set Acceptable &Image Types", self, triggered=self.setImageTypes)

    def createMenu(self):
        self.filemenu = self.menuBar().addMenu("&File")
        self.filemenu.addAction(self.openFolderAct)

        self.optionsmenu = self.menuBar().addMenu("&Options")
        self.optionsmenu.addAction(self.setWorkingDirAct)
        self.optionsmenu.addSeparator()
        self.optionsmenu.addAction(self.setTessDirAct)
        self.optionsmenu.addAction(self.setGSDirAct)
        self.optionsmenu.addAction(self.setImageTypesAct)

    def process(self):
        """
        This functions loops through every image to get the name it was assigned through the page_offset_func(),
        adds the image and name together as a tuple, and then sends it off for actual processing in the function
        process_images() (which is different than process_image()!!)
        """
        self.oldpointer = self.pointer
        image_and_names = []

        for index, im in enumerate(self.images):
            self.pointer = index
            if self.splitpageGB.isChecked():
                nameL, nameR = self.page_offset_func()
                image_tuple = (im, index, nameL, nameR)
            else:
                name = self.page_offset_func()
                image_tuple = (im, index, name)
            image_and_names.append(image_tuple)

        self.pointer = self.oldpointer

        self.mkdirectory(VAR.dirname_processed)

        if window.cropGB.isChecked():
            self.mkdirectory(VAR.dirname_cropped)

        worker = Worker(process_images, image_and_names)
        worker.signals.finished.connect(self.processing_finished)
        self.threadpool.start(worker)

    def processing_finished(self):
        self.processing_label.setText("")

    def ocr(self):
        """
        Threading examples came from:
        https://martinfitzpatrick.name/article/multithreading-pyqt-applications-with-qthreadpool/

        this function initiates the OCR workers
        """

        self.processing_label.setText("Starting OCR...")

        self.mkdirectory(VAR.dirname_ocr)

        # if process_images() was run, there will be a folder called VAR.dirname_processed, so get its files
        if os.path.isdir(VAR.dirname_processed):
            src_path = os.path.join(os.getcwd(), VAR.dirname_processed)
            self.final_images = [name for name in os.listdir(src_path) if
                                 name.lower().endswith(VAR.im_types) and name.startswith("Page")]
            preface_pages = [name for name in os.listdir(src_path) if
                             name.lower().endswith(VAR.im_types) and name.startswith("Preface")]
            # we got files that started with "Preface" separately and here we make sure they go at the beginning
            self.final_images = natsorted(preface_pages) + natsorted(self.final_images)

        # if process_images() was not run, we just get the images files from the current folder
        else:
            src_path = os.getcwd()

            self.final_images = [name for name in os.listdir(src_path) if
                                 name.lower().endswith(VAR.im_types)]
            # natsorted ensures that the images will come in the correct order even if numbers not zero-padded
            self.final_images = natsorted(self.final_images)

        # the counter marks progress. The worker threads don't know about each other, but they can increment this
        # counter in order to display what number they are out of the total
        self.counter = 0

        make_pdf = self.ocr_pdf_box.isChecked()
        make_txt = self.ocr_txt_box.isChecked()
        self.ocr_pdf_box.setEnabled(False)
        self.ocr_txt_box.setEnabled(False)
        for im in self.final_images:
            worker = Worker(OCRinternals, im, len(self.final_images), src_path, make_pdf, make_txt)
            worker.signals.result.connect(self.print_output)
            self.threadpool.start(worker)

    def print_output(self, s):
        self.processing_label.setText(s)

    def algorithm_changed(self):
        # when the algorithm is changed, enable or disable some sliders and change the threshold slider

        if self.algorithm.currentIndex() == 0:
            self.thres_min, self.thres_max, self.thres_slider.initval = 1, 255, 140
            if self.thres_slider.value() == 140:
                self.thres_slider.alg1val = -20
            else:
                self.thres_slider.alg1val = self.thres_slider.value()
            self.thres_slider.valuenum = self.thres_slider.alg0val
            self.boxsize_slider.setEnabled(False)
            self.boxsize_label.setEnabled(False)
            self.boxsize_spinbox.setEnabled(False)
        elif self.algorithm.currentIndex() == 1:
            self.thres_min, self.thres_max, self.thres_slider.initval = -100, 40, -20
            self.thres_slider.alg0val = self.thres_slider.value()
            self.thres_slider.valuenum = self.thres_slider.alg1val
            self.boxsize_slider.setEnabled(True)
            self.boxsize_label.setEnabled(True)
            self.boxsize_spinbox.setEnabled(True)
        self.thres_slider.setMinimum(self.thres_min)
        self.thres_slider.setMaximum(self.thres_max)
        self.thres_spinbox.setMinimum(self.thres_min)
        self.thres_spinbox.setMaximum(self.thres_max)
        self.thres_spinbox.setValue(self.thres_slider.valuenum)
        if hasattr(self, 'l_pix'):
            self.view_current()

    def imagebox_func(self):
        # enable image sensitivity controls if "look for images" box is checked
        state = self.algorithm_imagebox.isChecked()
        self.algorithm_imsens_label.setEnabled(state)
        self.algorithm_imsens.setEnabled(state)
        self.algorithm_imsens_val.setEnabled(state)
        self.view_current()

    def split_page_toggle(self):
        # if pages are split, then need two labels for L and R page, otherwise need only one label
        state = self.splitpageGB.isChecked()
        if state:
            self.rpage_label.show()
            self.row_viewer_labels.addWidget(self.rpage_label)
            self.drop_pageR_btn.setEnabled(True)
        else:
            self.rpage_label.hide()
            self.row_viewer_labels.removeWidget(self.rpage_label)
            self.drop_pageR_btn.setEnabled(False)
        self.view_current()

    def imsens(self):
        # sets image sensitivity
        self.algorithm_imsens_val.setText("{0:03d}".format(self.algorithm_imsens.value()))
        VAR.im_sens = self.algorithm_imsens.value()
        self.view_current()

    def cropbox_changed(self):
        # redraw screen when any of the crop boxes are changed
        if not self.this_page_only_check.isChecked():
            VAR.cropbox = (self.cropLbox.value(), self.cropTbox.value(), self.cropRbox.value(), self.cropBbox.value())
        else:
            self.local_adj[self.pointer]["cropbox"] = (self.cropLbox.value(), self.cropTbox.value(), self.cropRbox.value(), self.cropBbox.value())
            self.local_adj[self.pointer]["local"] = True
        self.view_current()

    def this_page_only_reset_func(self):
        button = QMessageBox.question(self, "Really reset?", "Are you sure you want to reset all individual adjustments?",
                             QMessageBox.Yes | QMessageBox.Cancel)
        if button == QMessageBox.Yes:
            for _, page in self.local_adj.items():
                page["offset"] = 0
                page["dropL"] = False
                page["dropR"] = False
                page["cropbox"] = (0, 0, 0, 0)
                page["local"] = False
            self.load_current()
        else:
            pass

    def this_page_only_func(self):
        if self.this_page_only_check.isChecked():
            self.cropGB.setStyleSheet("QSpinBox {border:1px solid red}")
            self.offset_slider.setStyleSheet("QSlider {border:1px solid red}")
            self.view.setStyleSheet("MyGraphicsView {border:3px solid red}")
        else:
            self.cropGB.setStyleSheet("QSpinBox {border:1px solid #cccccc}")
            self.offset_slider.setStyleSheet("QSlider {border:1px solid #f0f0f0}")
            self.view.setStyleSheet("MyGraphicsView {border:3px solid #828790}")
            if self.local_adj[self.pointer]["local"] == True:
                self.local_adj[self.pointer]["offset"] = 0
                self.local_adj[self.pointer]["cropbox"] = (0, 0, 0, 0)
                self.local_adj[self.pointer]["local"] = False
                self.load_current()

    def drop_pageL(self):
        state = self.drop_pageL_btn.isChecked()
        self.local_adj[self.pointer]["dropL"] = state
        self.view_current()

    def drop_pageR(self):
        state = self.drop_pageR_btn.isChecked()
        self.local_adj[self.pointer]["dropR"] = state
        self.view_current()

    def rotate(self):
        # set rotation
        VAR.rotation = self.rotate_box.value()
        self.view_current()

    def setThreshold(self):
        # set threshold from slider
        VAR.thres = self.thres_slider.value()
        self.thres_spinbox.setValue(VAR.thres)
        self.mini_process()

    def setThreshold_spinbox(self):
        # set threshold from spinbox
        if not self.thres_slider.isSliderDown():
            VAR.thres = self.thres_spinbox.value()
            self.thres_slider.setValue(VAR.thres)
            self.view_current()

    def setboxsize(self):
        # set boxsize from slider
        VAR.boxsize = self.boxsize_slider.value()
        self.boxsize_spinbox.setValue(VAR.boxsize)
        self.mini_process()

    def setboxsize_spinbox(self):
        # set boxsize from spinbox
        if not self.boxsize_slider.isSliderDown():
            VAR.boxsize = self.boxsize_spinbox.value()
            self.boxsize_slider.setValue(VAR.boxsize)
            self.view_current()

    def setOffset(self):
        # set split page offset
        if not self.this_page_only_check.isChecked():
            VAR.offset = self.offset_slider.value()
            self.offset_spinbox.setValue(VAR.offset)
        else:
            self.local_adj[self.pointer]["offset"] = self.offset_slider.value()
            self.offset_spinbox.setValue(self.local_adj[self.pointer]["offset"])
            self.local_adj[self.pointer]["local"] = True
        self.view_current()


    def openpdf(self):
        options = QFileDialog.Options()

        if VAR.setd['Last Directory']:
            default_folder = VAR.setd['Last Directory']
        else:
            default_folder = VAR.temp_base_dir

        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Open PDF File", default_folder,
                                                  "PDF Files (*.pdf)", options=options)
        self.progress_bar.setValue(0)
        self.dirlabel.setHidden(True)
        self.progress_bar.setHidden(False)
        if fileName:
            self.dirlabel.setText(fileName)
            self.load_pdf_workers(fileName)

    def load_pdf_workers(self, filename):

        dir, file = os.path.split(filename)
        fn, ext = os.path.splitext(file)

        VAR.setd['Last Directory'] = dir
        with open(VAR.settings_file, 'w') as f:
            VAR.settings.write(f)

        VAR.final_directory = dir
        VAR.filename = fn
        VAR.temp_dir = os.path.join(VAR.temp_base_dir, VAR.filename + "_BKtemp")
        output_fn = os.path.join(VAR.temp_dir, fn + "%03d.png")
        self.mkdirectory(VAR.temp_dir)

        # os.chdir(dir)

        worker = Worker(load_pdf, filename, output_fn)
        worker.signals.finished.connect(self.load_pdf_finished)
        worker.signals.progress.connect(self.load_pdf_progress)
        worker.signals.result.connect(self.set_view)
        self.directory_btn.setEnabled(False)
        self.threadpool.start(worker)

    def load_pdf_progress(self, progress):
        self.progress_bar.setValue(progress)

    def load_pdf_finished(self):
        self.progress_bar.setHidden(True)
        self.dirlabel.setHidden(False)
        self.directory_btn.setEnabled(True)

    def setExistingDirectory(self):
        options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly

        if VAR.setd['Last Directory']:
            default_folder = VAR.setd['Last Directory']
        else:
            default_folder = VAR.temp_base_dir

        working_directory = QFileDialog.getExistingDirectory(self,
                "Choose image directory...",
                default_folder, options=options)

        self.progress_bar.setValue(0)
        self.dirlabel.setHidden(True)
        self.progress_bar.setHidden(False)

        if working_directory:
            VAR.setd['Last Directory'] = working_directory
            with open(VAR.settings_file, 'w') as f:
                VAR.settings.write(f)

            os.chdir(working_directory)
            VAR.final_directory = os.getcwd()
            _, name = os.path.split(working_directory)
            VAR.filename = name
            VAR.temp_dir = os.path.join(VAR.temp_base_dir, VAR.filename + "_BKtemp")
            self.mkdirectory(VAR.temp_dir)
            self.dirlabel.setText(working_directory)
            self.load_image_workers()

    def load_image_workers(self):
        worker = Worker(load_images)
        worker.signals.finished.connect(self.load_images_finished)
        worker.signals.progress.connect(self.load_images_progress)
        worker.signals.result.connect(self.set_view)
        self.directory_btn.setEnabled(False)
        self.threadpool.start(worker)

    def load_images_progress(self, progress):
        self.progress_bar.setValue(progress)

    def load_images_finished(self):
        self.progress_bar.setHidden(True)
        self.dirlabel.setHidden(False)
        self.directory_btn.setEnabled(True)

    def set_view(self, result):
        os.chdir(VAR.temp_dir)
        self.images = result
        self.pointer = 0
        self.num_images = len(self.images)

        self.local_adj = {}
        for i in range(self.num_images):
            self.local_adj[i] = {
                "local": False,
                "dropL": False,
                "dropR": False,
                "offset": 0,
                "cropbox": (0, 0, 0, 0)
            }

        self.zoom_step = 0.1
        self.algorithm_changed()
        for control in self.name_controls + self.bottom_controls:
            control.setEnabled(True)
        self.splitpageGB.setEnabled(True)
        self.cropGB.setEnabled(True)
        self.processGB.setEnabled(True)

        self.w_vsize = self.view.size().width()
        self.h_vsize = self.view.size().height()
        if self.w_vsize <= self.h_vsize:
            self.max_vsize = self.w_vsize
        else:
            self.max_vsize = self.h_vsize

        self.load_current()
        self.imagebox_func()

        self.name_input.setText(VAR.filename)
        self.viewer_scrollbar.setMinimum(0)
        self.viewer_scrollbar.setMaximum(self.num_images - 1)
        self.gotopage_box.setMinimum(1)
        self.gotopage_box.setMaximum(self.num_images)
        self.gotopage_labelR.setText("of {}".format(self.num_images))

    def next(self):
        self.pointer = (self.pointer + 1) % self.num_images
        self.viewer_scrollbar.setValue(self.pointer)

    def prev(self):
        self.pointer = (self.pointer - 1) % self.num_images
        if self.pointer < 0:
            self.pointer += self.num_images
        self.viewer_scrollbar.setValue(self.pointer)

    def viewer_slider_moved(self):
        self.pointer = self.viewer_scrollbar.value()
        self.gotopage_box.setValue(self.pointer + 1)
        self.load_current()

    def viewer_slider_released(self):
        self.load_current()

    def viewer_box_changed(self):
        if not self.viewer_scrollbar.isSliderDown():
            self.pointer = self.gotopage_box.value() - 1
            self.viewer_scrollbar.setValue(self.pointer)

    def mini_process(self):
        if self.splitpageGB.isChecked():
            self.c_viewLRorig = split_image(self.l_pix, pointer=self.pointer)
        else:
            self.c_viewLRorig = [self.l_pix]

        if not hasattr(self, 'scaleheight'):
            self.scaleheight = self.max_vsize


        if self.scaleheight > (self.c_viewLRorig[0].height / 2):
            self.scaleheight_preview = self.c_viewLRorig[0].height / 2
        else:
            self.scaleheight_preview = self.scaleheight

        ratio = self.scaleheight_preview / self.c_viewLRorig[0].height

        if not self.local_adj[self.pointer]["dropL"]:
            self.c_viewLRorig[0] = process_image(self.algorithm.currentIndex(), self.algorithm_imagebox.isChecked(),
                                               scale_pil(self.c_viewLRorig[0], self.scaleheight_preview), ratio=ratio)
        else:
            self.c_viewLRorig[0] = process_dropped(scale_pil(self.c_viewLRorig[0], self.scaleheight))

        if len(self.c_viewLRorig) == 2:
            if not self.local_adj[self.pointer]["dropR"]:
                self.c_viewLRorig[1] = process_image(self.algorithm.currentIndex(), self.algorithm_imagebox.isChecked(),
                                                     scale_pil(self.c_viewLRorig[1], self.scaleheight_preview), ratio=ratio)
            else:
                self.c_viewLRorig[1] = process_dropped(scale_pil(self.c_viewLRorig[1], self.scaleheight))

        if self.scaleheight_preview != self.scaleheight:
            self.c_viewLRorig = [scale_pil(x, self.scaleheight) for x in self.c_viewLRorig]

        self.render_current()

    def view_current(self):
        if self.splitpageGB.isChecked():
            self.c_viewLRorig = split_image(self.l_pix, pointer=self.pointer)
        else:
            self.c_viewLRorig = [self.l_pix]

        if not hasattr(self, 'scaleheight'):
            self.scaleheight = self.max_vsize


        if not self.local_adj[self.pointer]["dropL"]:
            self.c_viewLRorig[0] = scale_pil(process_image(self.algorithm.currentIndex(), self.algorithm_imagebox.isChecked(),
                                                           self.c_viewLRorig[0]), self.scaleheight)
        else:
            self.c_viewLRorig[0] = scale_pil(process_dropped(self.c_viewLRorig[0]), self.scaleheight)

        if len(self.c_viewLRorig) == 2:
            if not self.local_adj[self.pointer]["dropR"]:
                self.c_viewLRorig[1] = scale_pil(process_image(self.algorithm.currentIndex(), self.algorithm_imagebox.isChecked(),
                                                               self.c_viewLRorig[1]), self.scaleheight)
            else:
                self.c_viewLRorig[1] = scale_pil(process_dropped(self.c_viewLRorig[1]), self.scaleheight)

        self.render_current()

    def render_current(self):

        if len(self.c_viewLRorig) == 2:
            scene_width = self.c_viewLRorig[0].width + self.c_viewLRorig[1].width
        else:
            scene_width = self.c_viewLRorig[0].width

        self.scene.clear()
        self.scene.setSceneRect(0, 0, scene_width, self.c_viewLRorig[0].height)
        pix1 = QGraphicsPixmapItem(ImageQt.toqpixmap(self.c_viewLRorig[0]))
        pix1.setPos(0, 0)

        self.scene.addItem(pix1)

        if len(self.c_viewLRorig) == 2:
            pix2 = QGraphicsPixmapItem(ImageQt.toqpixmap(self.c_viewLRorig[1]))
            pix2.setPos(self.c_viewLRorig[0].width + 10, 0)
            self.scene.addItem(pix2)

    def page_offset_func(self):
        poffset = self.page_input.value()

        if self.splitpageGB.isChecked():
            Lnum = self.pointer * 2 - poffset
            Rnum = self.pointer * 2 + 1 - poffset

            if Lnum > 0:
                return "Page {:03d}".format(Lnum), "Page {:03d}".format(Rnum)
            elif Lnum == 0 and Rnum + poffset == 1:
                return "[Not Used]", "Page {0:03d}".format(self.pointer * 2 + 1)
            elif Lnum == 0 and Rnum + poffset != 1:
                return "Preface Page {0:02d}".format(poffset), "Page {0:03d}".format(Rnum)
            elif Lnum < 0:
                return "Preface Page {0:02d}".format(poffset - abs(Lnum)), "Preface Page {0:02d}".format(poffset - abs(Rnum))
        else:
            num = self.pointer - poffset + 1
            if num > 0:
                return "Page {0:03d}".format(num)
            elif num <= 0:
                return "Preface Page {0:02d}".format(poffset - abs(num))

    def load_current(self):
        self.l_pix = self.images[self.pointer]

        self.update_page_offset()

        local_adj = self.local_adj[self.pointer]

        if self.local_adj[self.pointer]["local"]:  #Adjust parameters taking into account local values if the "Local" flag is True
            self.this_page_only_check.setChecked(True)
            self.this_page_only_func()

            self.offset_slider.blockSignals(True)
            self.offset_slider.setValue(local_adj["offset"])
            self.offset_spinbox.setValue(local_adj["offset"])
            self.offset_slider.blockSignals(False)
            
            for box in self.cropboxes:
                box.blockSignals(True)
            self.cropLbox.setValue(local_adj["cropbox"][0])
            self.cropTbox.setValue(local_adj["cropbox"][1])
            self.cropRbox.setValue(local_adj["cropbox"][2])
            self.cropBbox.setValue(local_adj["cropbox"][3])
            for box in self.cropboxes:
                box.blockSignals(False)

        else:
            self.this_page_only_check.setChecked(False)
            self.this_page_only_func()

            self.offset_slider.blockSignals(True)
            self.offset_slider.setValue(VAR.offset)
            self.offset_spinbox.setValue(VAR.offset)
            self.offset_slider.blockSignals(False)

            for box in self.cropboxes:
                box.blockSignals(True)
            self.cropLbox.setValue(VAR.cropbox[0])
            self.cropTbox.setValue(VAR.cropbox[1])
            self.cropRbox.setValue(VAR.cropbox[2])
            self.cropBbox.setValue(VAR.cropbox[3])
            for box in self.cropboxes:
                box.blockSignals(False)

        self.drop_pageL_btn.blockSignals(True)
        self.drop_pageR_btn.blockSignals(True)
        self.drop_pageL_btn.setChecked(local_adj["dropL"])
        self.drop_pageR_btn.setChecked(local_adj["dropR"])
        self.drop_pageL_btn.blockSignals(False)
        self.drop_pageR_btn.blockSignals(False)

        # Use the quick preview if scrolling, else use normal view
        if self.viewer_scrollbar.isSliderDown():
            self.mini_process()
        else:
            self.view_current()

    def update_page_offset(self):
        bookname = self.name_input.text()

        if self.splitpageGB.isChecked():
            Ln, Rn = self.page_offset_func()
            Ln, Rn = bookname + " " + Ln, bookname + " " + Rn

            self.lpage_label.setText(Ln)
            self.rpage_label.setText(Rn)
        else:
            Ln = self.page_offset_func()
            Ln = bookname + " " + Ln
            self.lpage_label.setText(Ln)

    def wheel_event(self, event):
        numDegrees = event.angleDelta() / 8
        numSteps = numDegrees / 15.0
        self.zoom(numSteps)
        event.accept()

    def zoom(self, step):
        self.scene.clear()

        self.scaleheight = self.c_viewLRorig[0].height * (1 + self.zoom_step * step.y())

        self.view_current()

    def mkdirectory(self, dirname):
        """
        Makes a new directory with name dirname, or clears that directory if it already exists
        """

        errors = []

        def del_files_in_folder(dir):
            for file in os.listdir(dir):
                file = os.path.join(dir, file)
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                    elif os.path.isdir(file):
                        del_files_in_folder(file)
                        os.rmdir(file)
                except OSError:
                    errors.append(file)
            return

        try:
            os.makedirs(dirname)
        except FileExistsError:
            del_files_in_folder(dirname)

        if errors:
            msgText = "Warning: the following files or folders were unable to be cleared:\n\n"
            for i, error in enumerate(errors):
                msgText += "{}: {}\n".format(i+1, error)
            msgText += "\nYou can ignore these, but it might mean that the editor will be opening up obsolete files"

            QMessageBox.information(self, "Overwrite Errors",
                                    msgText, QMessageBox.Ok)

        return



class MyGraphicsView(QGraphicsView):
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            super(MyGraphicsView, self).wheelEvent(event)
        else:
            window.wheelEvent(event)


class MySlider(QSlider):
    def mouseDoubleClickEvent(self, event):
        super(MySlider, self).mouseDoubleClickEvent(event)
        self.setValue(self.initval)

    def mouseMoveEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if not window.this_page_only_check.isChecked():
                window.this_page_only_check.setChecked(True)
                window.this_page_only_func()
            super(MySlider, self).mouseMoveEvent(event)
        else:
            super(MySlider, self).mouseMoveEvent(event)


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


def process_images(im_and_names, progress_callback):

    if window.splitpageGB.isChecked():
        for im, pointer, nameL, nameR in im_and_names:

            im_procLR = split_image(im, pointer=pointer)

            if window.processGB.isChecked():
                im_procLR = [process_image(window.algorithm.currentIndex(), window.algorithm_imagebox.isChecked(), x, final=True)
                           for x in im_procLR]

            if not window.local_adj[pointer]["dropL"]:
                im_procLR[0].save(os.path.join(VAR.dirname_processed, nameL + ".png"))
            if not window.local_adj[pointer]["dropR"]:
                im_procLR[1].save(os.path.join(VAR.dirname_processed, nameR + ".png"))

            if window.cropGB.isChecked():
                im_procLR = [final_crop(x, pointer=pointer) for x in im_procLR]

                if not window.local_adj[pointer]["dropL"]:
                    im_procLR[0].save(os.path.join(VAR.dirname_cropped, nameL + ".png"))
                if not window.local_adj[pointer]["dropR"]:
                    im_procLR[1].save(os.path.join(VAR.dirname_cropped, nameR + ".png"))

            window.processing_label.setText("Processing pages: {} and {}".format(nameL, nameR))

    else:
        for im, pointer, name in im_and_names:
            if window.processGB.isChecked():
                im_proc = process_image(window.algorithm.currentIndex(), window.algorithm_imagebox.isChecked(), im, final=True)
            else:
                im_proc = im

            if not window.local_adj[pointer]["dropL"]:
                im_proc.save(os.path.join(VAR.dirname_processed, name + ".png"))

            if window.cropGB.isChecked():
                im_proc = final_crop(im, pointer=pointer)
                if not window.local_adj[pointer]["dropL"]:
                    im_proc.save(os.path.join(VAR.dirname_cropped, name + ".png"))

            window.processing_label.setText("Processing page: {}".format(name))

    return


def OCRinternals(im, total, src_path, make_pdf, make_txt, progress_callback):
    f, e = os.path.splitext(im)
    old_file = os.path.join(src_path, im)
    new_file = os.path.join(os.getcwd(), VAR.dirname_ocr, f)
    if os.path.isdir(VAR.dirname_cropped):
        old_file_txt = os.path.join(os.getcwd(), VAR.dirname_cropped, im)
    else:
        old_file_txt = old_file
    new_file_txt = os.path.join(os.getcwd(), VAR.dirname_ocr, f + '.txt')

    if make_pdf:
        subprocess.call([VAR.path_to_tess, '--oem', '1', old_file, new_file, '-l', 'eng', 'pdf'])

    if make_txt:
        subprocess.call([VAR.path_to_tess, '--oem', '1', old_file_txt, new_file, '-l', 'eng'])

        with open(new_file_txt, 'r', encoding='utf-8') as txtfiletemp:
            text = txtfiletemp.read()

        text = regex.sub(r"^\n+|\n+$", "", text)  # take out leading and trailing newlines
        text = regex.sub(r"[-—]\n", "", text)  # take out broken words
        text = regex.sub(r"\n\n(?=[a-z])", " ", text)  # take out unnecessary double \n
        text = regex.sub(r"(?<!\n)\n(?!\n)", " ", text)  # take out unnecessary single \n
        text = regex.sub(r"“|”", "\"", text)  # replace double quotations
        text = regex.sub(r"‘|’", "\'", text)  # replace single quotations
        text = regex.sub(r"—", "-", text)  # replace dashes
        text = '\n\n' + '>>> ' + f + '\n\n' + text

        with open(new_file_txt, 'w', encoding='utf-8') as txtfiletemp:
            txtfiletemp.write(text)

    counter = window.counter
    window.counter += 1

    if window.counter == total:
        window.processing_label.setText("OCR finished. Merging files...")
        mergeOCR(make_pdf, make_txt)
        window.ocr_pdf_box.setEnabled(True)
        window.ocr_txt_box.setEnabled(True)
        return "All Complete!"
    else:
        return "Completed OCR for {} ({} of {})".format(f, counter, total)


def mergeOCR(make_pdf, make_txt, name='default'):

    if window.name_input.text():
        name = window.name_input.text()

    if make_pdf:
        os.chdir(VAR.dirname_ocr)
        pdffile_list = [os.path.splitext(x)[0] + ".pdf" for x in window.final_images]

        merger = PyPDF2.PdfFileMerger()
        for file in pdffile_list:
            with open(file, 'rb') as pdffile:
                merger.append(PyPDF2.PdfFileReader(pdffile))
        os.chdir("..")
        try:
            merger.write(os.path.join(VAR.final_directory, name + '.pdf'))
        except FileExistsError:
            i = 0
            while os.path.exists(os.path.join(VAR.final_directory, name + '{}.pdf'.format(i))):
                i += 1
            merger.write(os.path.join(VAR.final_directory, name + '{}.pdf'.format(i)))

    if make_txt:
        os.chdir(VAR.dirname_ocr)
        txtfile_list = [os.path.splitext(x)[0] + ".txt" for x in window.final_images]

        txtfile_path_final = os.path.join(os.getcwd(), name + '.txt')
        with open(txtfile_path_final, 'w', encoding='utf-8') as txtfile:
            txtfile.write('Begin Book\n\n')

        for file in txtfile_list:
            with open(file, 'r', encoding='utf-8') as in_file:
                text = in_file.read()
            with open(txtfile_path_final, 'a', encoding='utf-8') as txtfile:
                txtfile.write(text)
        os.chdir("..")
        try:
            os.rename(txtfile_path_final, os.path.join(VAR.final_directory, name + '.txt'))
        except FileExistsError:
            i = 0
            while os.path.exists(os.path.join(VAR.final_directory, name + '{}.txt'.format(i))):
                i += 1
            os.rename(txtfile_path_final, os.path.join(VAR.final_directory, name + '{}.txt'.format(i)))

    return





# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook

def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Converter()
    window.show()
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")
