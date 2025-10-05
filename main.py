from openocr import OpenOCR
engine = OpenOCR()
img_path = 'test_image.png'
result, elapse = engine(img_path)