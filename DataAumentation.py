import albumentations as A
import cv2
import os


transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45)
])

PATH = r'C:\Users\Bryan\Downloads\Colores'
DIR = 'Yellow'
OUT_PATH = r'C:\Users\Bryan\Downloads\Colores\Yellow'

c2 = 0
# for category in CATEGORIES:
path = os.path.join(PATH, DIR)
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img))
    count = 0
    while (count < 4):
        augmented_image = transform(image=img_array)['image']
        Datos = '{}/aum_{}_{}.jpg'.format(OUT_PATH, count, c2)
        cv2.imwrite(Datos, augmented_image)
        count += 1
        c2 += 1
print('total img =', c2)