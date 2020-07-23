import os

DATA_PATH = '..\\Data\\RGB'

IMG_NAME_LIST = [f for dp, dn, filenames in os.walk(DATA_PATH) for f in filenames if os.path.splitext(f)[1].lower() == '.jpg']
IMGNAME = os.path.splitext(IMG_NAME_LIST[0])[0]
print(IMG_NAME_LIST)
print(IMGNAME)
new = os.path.join(DATA_PATH, IMGNAME)
print(new)