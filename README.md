# BK-Converter

## What is it?
BK-Converter (short for "Book Converter") was primarily written to compile a single book-like txt file from a non-DRM protected PDF, allowing one to more conveniently read the text of a PDF article on an e-reader. It can also convert a series of image files into a book-like txt file. After pointing BK-Converter to a PDF file (or a directory containing a series of image files), you can set the correct page numbering, split the pages into left and right pages, crop out certain elements of the page, and process the image to improve readability (and OCR-ibility). The output txt file is delineated by page number, and ensures that paragraphs are not interspered with newlines (as they often are with direct text output from a PDF).

## How does it work
BK-Converter is a PyQt5 interface for Ghostscript (for converting PDFs to images) and Tesseract (for doing OCR on those images). It uses Pillow and NumPy for image processing. The txt output is cleaned up considerably through the use of regex.

