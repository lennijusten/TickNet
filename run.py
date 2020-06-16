from ftplib import FTP_TLS

import gspread
from PIL import Image
from math import floor
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# ACTUAL (Tick App)
# todo 1) Improve model
# todo 2) Clean up log file print commands
# todo 3) Start thinking about errors that could occur or exceptions to include
# Crop a custom batch of images
# Predict a single image or custom batch
# todo 4) Map class labels to categorical classes and print to google sheet
# How to implement four probabilities for all the classes
# Change log for google sheets?
# What predictions are preset?
# Update sheet with each new run so that incorrect predictions are colored red and correct predictions are colored
# green
# Keep a network statistics sheet with accuracy, confusion matrix, etc
# Save matlab read-able filenames in google sheets
# todo 5) Clarify storage protocol for cropped images and filenames


username = 'ljusten@wisc.edu'
password = 'Germany2013!_m'
service_acc = '/Users/Lenni/Documents/PycharmProjects/MCEVBD Fellowship/service_account.json'

ss = "Copy of Image_ID"
save_path = '/Users/Lenni/Desktop/Box Images/'
crop_path = '/Users/Lenni/Desktop/Box Images Cropped/'
# box_path = 'WMEL-Tick Images/Submitted Images'
box_path = 'tick-id-images'

labels = ['Dermacentor_a_m_unfed', 'Ixodes_a_f_unfed', 'Dermacentor_a_f_unfed', 'Ixodes_a_m_unfed']
img_height = 224
img_width = 224

ftp = FTP_TLS('ftp.box.com')
print("Accessing box.com with login credentials...")
print("Welcome: ", ftp.getwelcome())
ftp.login(user=username, passwd=password)
print("Log-in successful. Navigating to ", box_path)

main_dir = ftp.retrlines('LIST')
ftp.pwd()
ftp.cwd(box_path)

filenames = ftp.nlst()  # get filenames within the directory
filenames.remove('.')
filenames.remove('..')
print(filenames)

print("Found {} images in folder. Retrieving...".format(len(filenames)))
new_count = 0
exist_count = 0

new_files = []
new_filenames = []

exist_files = []
exist_filenames = []

id = []
for filename in filenames[0:10]:
    local_filename = os.path.join(save_path, filename)
    if not os.path.exists(local_filename):
        new_count += 1
        file = open(local_filename, 'wb')
        ftp.retrbinary('RETR ' + filename, file.write)
        new_files.append(local_filename)
        new_filenames.append(filename)
        id.append(os.path.splitext(filename)[0])
        file.close()
    else:
        exist_count += 1
        exist_files.append(local_filename)
        exist_filenames.append(filename)
        id.append(os.path.splitext(filename)[0])

ftp.quit()  # close connection
print(new_count, " new images were downloaded")
print(exist_count, " images already existed in local directory:", save_path)


def im_crop(files, destination):
    crop_count = 0
    new_crop_imgs = []
    for filename in files:
        im = Image.open(filename)

        dim = im.size
        shortest = min(dim[0:2])
        longest = max(dim[0:2])

        lv = np.array(range(0, shortest)) + floor((longest - shortest) / 2)
        if dim[0] == shortest:
            im_cropped = np.asarray(im)[lv, :, :]
        else:
            im_cropped = np.asarray(im)[:, lv, :]

        im_cropped = Image.fromarray(im_cropped)
        im_cropped.save(os.path.join(destination, os.path.basename(im.filename)))
        new_crop_imgs.append(os.path.join(destination, os.path.basename(im.filename)))
        crop_count += 1

    print('{} Images cropped'.format(crop_count))
    return new_crop_imgs


print("Cropping images and saving to ", crop_path)
print("Cropping successful")
# _cropped_imgs = im_crop(new_files, crop_path)
cropped_imgs = im_crop(exist_files, crop_path)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# file_path = '/Users/Lenni/Documents/Independent Research Project/Data/Collection 9 Cropped/Ixodes_scapularis_f_a_tr_unk_unfed_032320_2.JPG'

images = []
for fname in cropped_imgs:
    img = image.load_img(fname, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)
results = loaded_model.predict(images, batch_size=10)
class_prob = np.amax(results, 1).tolist()
classes = np.argmax(results, 1)

tabs = []
species = []
sex = []
for idx in classes:
    species.append(labels[idx].split('_')[0])
    sex.append(labels[idx].split('_')[2].capitalize())
    tabs.append(labels[idx].split('_'))

input = [species, sex, class_prob]

gc = gspread.service_account(filename=service_acc)
sh = gc.open(ss)
worksheet = sh.sheet1
column_headers = worksheet.row_values(1)
print(column_headers)

try:
    NN_Prob = worksheet.find("NN Prob")
    NN_Species = worksheet.find("NN Species")
    NN_Sex = worksheet.find("NN Sex")
    cols = [NN_Species, NN_Sex, NN_Prob]
except gspread.exceptions.CellNotFound as err:
    print("Column name '{}' was not found in Row 1".format(err))

for i in range(len(id)):
    rows = worksheet.findall(id[i], in_column=worksheet.find("Tick_ID").col)
    if len(rows) == 0:
        print("Tick ID '{}' was not found".format(id[i]))
    elif len(rows) > 1:
        print("Multiple rows with Tick ID '{}' found. Copying TickNet results to all rows...".format(id[i]))
        print("Rows: ".format(rows))
    else:
        print("Tick ID {} found. Copying TickNet results to row...".format(id[i]))
        pass

    for r in rows:
        for c in range(len(cols)):
            worksheet.update_cell(r.row, cols[c].col, input[c][i])
        print("Row: ", r.row)