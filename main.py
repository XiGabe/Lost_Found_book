from openocr import OpenOCR
engine = OpenOCR()
img_path = './data/dataset1/8.png'
result, elapse = engine(img_path)