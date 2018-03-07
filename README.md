# BK-Converter

## What is it?
I have an obsession with reading things on my e-book, so BK-Converter (short for "Book Converter") was written to compile a single book-like plain text file from a PDF or a series of image files.

## Features
* Import PDFs or point BK-Converter to a folder containing image files
* Split a single image into left and right pages
* Process the images into black and white for increased readibility/ocr-ability. Can control the split point for black and white. Also included is a useful "box algorithm" which modifies the split point based on the average luminosity of a small box of pixels- very useful for images which are light in some areas but dark in others.
* Can view the results of any processing in real time in the application window
* Can crop images (only affects OCR output right now)
* Can control certain parameters separately per-page (for left/right split and cropping only right now)
* Tesseract OCR - can make a searchable PDF of output and/or txt file
* Under-the-hood clean-up of OCR output, so that paragraphs are separately properly and superfluous newlines are taken out, among other things
* Multithreaded OCR processing

## How does it work
BK-Converter is a PyQt5 interface for Ghostscript (for converting PDFs to images) and Tesseract (for doing OCR on those images). It uses Pillow and NumPy for image processing. The txt output is cleaned up considerably through the use of regex.

## Things to add:
* Need to add some way to interrupt/pause OCR
