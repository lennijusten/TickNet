import argparse
import datetime
import os
from ftplib import FTP_TLS
import glob
import shutil
from tensorflow.keras.models import model_from_json
import numpy as np
from math import floor
from PIL import Image
import pandas as pd
import gspread
from tqdm import tqdm
import getpass
import logging
import sys
import csv

# todo Send relevant people service_account json file and add instructions on where to move it
# todo Check WMEL file paths
# todo Update entry_dict, service account, spreadsheet, and classes

cwd = os.getcwd()

classes = ['Amblyomma americanum', 'Dermacentor variabilis', 'Ixodes scapularis']
ss = 'Copy of Image_ID'
service_acc = '/Users/Lenni/Documents/PycharmProjects/MCEVBD Fellowship/GH Rep/service_account.json'

entry_dict = {
    "Species": ['Dermacentor variabilis', 'Dermacentor andersoni',
                'Ixodes scapularis',
                'Amblyomma americanum', 'Amblyomma maculatum'],
    "Sex": ['M', 'F'],
    "Lifestage": ['Adult', 'Nymph', 'Larvae'],
    "Fed": ['yes', 'no']
}


def read_args():  # Get user arguments
    now = datetime.datetime.now()

    parser = argparse.ArgumentParser(description="Tick App prediction node")

    parser.add_argument("--mode",
                        type=str.lower,
                        choices=['predict', 'download', 'sync', 'auto'],
                        default=None,
                        help="Run mode")

    parser.add_argument("--source",
                        type=str.lower,
                        choices=['box', 'disk'],
                        default=None,
                        help="File source")

    parser.add_argument("--path",
                        type=str,
                        default=None,
                        help="Mode specific path")

    parser.add_argument("--model_dir",
                        type=str,
                        default='Model',  # todo add default model path
                        help="path to model dir with .json and .h5 files")

    parser.add_argument("--output",
                        type=str,
                        default='output',  # todo check default output path
                        help="output directory")

    parser.add_argument("--check_path",
                        type=str,
                        default=None,
                        help="Check path for existing images (only active when mode=download)")

    parser.add_argument("--destination",
                        type=str,
                        default=None,
                        help="Image destination (only active when mode=download)")

    parser.add_argument("--write",
                        action="store_true",
                        help="write results to spreadsheet")

    parser.add_argument("--year",
                        type=int,
                        default=None,
                        help="singular year from which to sync data")

    parser.add_argument("--WMEL_path",
                        type=str.lower,
                        default=None,
                        help="Path to WMEL file structure")

    parser.add_argument("--disk_transfer",
                        action="store_true",
                        help="Copy disk-files to output dir (only for mode='auto' with source='disk')")

    parser.add_argument("--logging",
                        action="store_true",
                        help="Specify for more detailed logging (helpful for debugging)")

    args = parser.parse_args()

    return args

args = read_args()

WMEL_raw_path = "/Volumes/WMEL/Tick Images/NN Images/Raw (if exists)/Tick ID images {}".format(args.year)
WMEL_renamed_path = "/Volumes/WMEL/Tick Images/NN Images/Renamed (all)/Tick ID images {}".format(args.year)

if args.logging:
    logging.basicConfig(filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
else:
    logging.basicConfig(filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')

# todo Return configuration files that lists the arguments

def ftp_login():  # Login to box server
    logging.info("ftp_login() -- Attempting BOX log-in")

    username = input("BOX username: ")  # Get user input
    password = getpass.getpass(prompt="BOX password:")  # Get user password (secure)

    ftp = FTP_TLS('ftp.box.com')
    print("Accessing box.com with login credentials...")
    ftp.login(user=username, passwd=password)
    print("Success")

    logging.info("ftp_login() -- Success")
    return ftp


def ftp_navigate(ftp, path):  # navigate to box path
    logging.info("ftp_navigate() -- Navigating to {}".format(path))
    ftp.cwd(path)
    logging.info("ftp_navigate() -- Success")
    return ftp


def box_download(ftp, destination, check_dir):  # Download box files with ftp connection (slow)
    logging.info("box_download() -- check_dir = {}, destination = {}".format(check_dir, destination))

    box_files = ftp.nlst()  # get filenames within the directory
    box_files.remove('.')
    box_files.remove('..')

    if check_dir is None:
        new_files = box_files
        print("{} images in BOX dir".format(len(new_files)))
        logging.info("box_download() -- {} images in BOX dir".format(len(new_files)))
    else:
        existing_files = []
        new_files = []
        for f in box_files:  # Check if files exist in check_dir
            if glob.glob(os.path.join(check_dir, os.path.splitext(f)[0] + ".*")):
                existing_files.append(f)
            else:
                new_files.append(f)

        print("{} new images, {} existing images on the local disk, {} total images in BOX dir".
              format(len(new_files), len(existing_files), len(box_files)))

        logging.info("box_download() -- {} new images, {} existing images on the local disk, {} total images on disk".
                     format(len(new_files), len(existing_files), len(box_files)))

    if new_files:  # Retrieve new images and write to disk
        print("Retrieving new images...")
        logging.warning("FTP download is very slow. "
                        "For a large number of images it is recommended to use the web browser to download files.")

        with open(os.path.join(args.output, "box-files.csv"), 'w') as myfile:  # todo test
            wr = csv.writer(myfile)
            wr.writerow(["fname"])
            for f in tqdm(new_files):
                filename = os.path.join(destination, f)
                wr.writerow([f])

                img = open(filename, 'wb')
                ftp.retrbinary('RETR ' + f, img.write)
                img.close()

    ftp.quit()  # close connection

    logging.info("box_download() -- ftp connection closed")
    logging.info("box_download() -- Success")
    return new_files


def get_input(path):  # Load image
    img = Image.open(path)
    return img


def im_crop(image):  # Crop images into a square along their shortest side
    dim = image.size
    shortest = min(dim[0:2])
    longest = max(dim[0:2])

    if shortest != longest:
        lv = np.array(range(0, shortest)) + floor((longest - shortest) / 2)
        if dim[0] == shortest:
            im_cropped = np.asarray(image)[lv, :, :]
        else:
            im_cropped = np.asarray(image)[:, lv, :]

        im_cropped = Image.fromarray(im_cropped)
    else:
        im_cropped = image

    return im_cropped


def matchExtensions(dir, extension):  # standardize image extensions (.JPG, .jpeg, .png, .PNG, etc)
    logging.info("matchExtensions() -- dir = {}, extension = {}".format(dir, extension))
    for filename in os.listdir(dir):
        infilename = os.path.join(dir, filename)
        if not os.path.isfile(infilename): continue
        base = os.path.splitext(infilename)[0]
        os.rename(infilename, base + extension)
    logging.info("matchExtensions() -- Success")


def loadModel(path):  # Load neural network with saved weights and architecture
    logging.info("loadModel() -- Loading model from {}".format(path))

    json_path = glob.glob(os.path.join(path, "*.json"))  # look for any file with .json extension
    weights_path = glob.glob(os.path.join(path, "*.h5"))  # look for any file with .h5 extension
    logging.info("loadModel() -- json_path = {}, weights_path = {}".format(json_path, weights_path))

    if len(json_path) > 1 or len(weights_path) > 1:
        print("WARNING: More than one instance of the model/weights was found. Selecting {}, {}.".
              format(json_path[0], weights_path[0]))
        logging.warning("loadModel() -- More than one instance of models/weights was found. "
                        "Selecting first instance: {}, {}".format(json_path[0], weights_path[0]))

    json_file = open(json_path[0], 'r')  # load first model instance
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path[0])  # load first weights instance

    print("Model loaded from disk.")
    logging.info("loadModel() -- Success")
    return loaded_model


def NNPredict(model, path, filenames, classes, err_f, err_r):  # Make predictions
    logging.info("NNPredict() -- loading images {} images".format(len(filenames)))
    print("Loading images...")
    input_shape = model.layers[0].input_shape  # get required image input size (model specific)
    logging.info("NNPredict() -- input_shape = {}".format(input_shape))

    fname = []
    tick_id = []
    images = []
    temp_err = []
    c = 0
    for f in tqdm(filenames):
        try:
            img = get_input(os.path.join(path, f))
            img_cropped = im_crop(img)
            img_resized = img_cropped.resize((input_shape[1], input_shape[2]))
            pixels = np.asarray(img_resized)  # convert image to array
            pixels = pixels.astype('float32')
            pixels /= 255.0  # rescale image from 0-255 to 0-1 (best practice)
            input = np.expand_dims(pixels, axis=0)  # adds batch dimension
            images.append(input)

            tick_id.append(os.path.splitext(f)[0])
            fname.append(f)
            c += 1
        except:
            err_f.append(f)
            err_r.append("Image could not be opened")
            temp_err.append(f)
            pass

    for f in temp_err:
        logging.warning("NNPredict() -- {} could not be opened.".format(f))

    # stack up images list to pass for prediction
    images = np.vstack(images)

    logging.info("NNPredict() -- {} images loaded. Initializing model.predict...".format(c))
    print("Evaluating images...")
    results = model.predict(images, batch_size=32, verbose=1)

    class_prob = np.amax(results, 1).tolist()
    rounded_class_prob = [round(100 * x, 2) for x in class_prob]
    class_ind = np.argmax(results, 1)
    preds = [classes[i] for i in class_ind]

    NN_dict = {
        "fname": fname,
        "tick_id": tick_id,
        "prediction": preds,
        "class": class_ind.tolist(),
        "prob": rounded_class_prob,
        "results": results.tolist()
    }

    df = pd.DataFrame(NN_dict)
    logging.info("NNPredict() -- Success")
    return df, err_f, err_r


def initSheet(service_acc, ss, year):  # Connect to google sheets api with service account
    logging.info("initSheet() -- Initializing sheet with service_acc = {}, ss = {}, year = {}".
                 format(service_acc, ss, year))
    print("Connecting to Google Sheets {} with login credentials...".format(ss))
    gc = gspread.service_account(filename=service_acc)
    sh = gc.open(ss)
    print("Opening {} sheet...".format(str(year)))
    worksheet = sh.worksheet(str(year))
    print("Success")
    logging.info("initSheet() -- Success")
    return worksheet


def write_to_sheet(worksheet, results, err_f, err_r):  # Write results to google sheets with gspread batch api
    logging.info("write_to_sheet() -- Initializing with {} results".format(len(results)))
    print("Writing results to worksheet...")
    col_names = worksheet.row_values(1)
    n_rows = worksheet.row_count
    logging.info("write_to_sheet() -- col_names = {}".format(col_names))
    logging.info("write_to_sheet() -- initial row count = {}".format(n_rows))

    ID_col = chr(ord('@') + col_names.index("Tick_ID") + 1)
    ID_range = "{}2:{}{}".format(ID_col, ID_col, n_rows)

    ss_id = worksheet.batch_get([ID_range])
    ss_id = [item for sublist in ss_id for item in sublist]  # Flatten list of lists

    n_rows = len(ss_id)  # Discount empty rows
    logging.info("write_to_sheet() -- row count = {}".format(n_rows))

    Species_col = chr(ord('@') + col_names.index("NN Species") + 1)
    Species_range = "{}2:{}{}".format(Species_col, Species_col, n_rows+1)

    Prob_col = chr(ord('@') + col_names.index("NN Prob") + 1)
    Prob_range = "{}2:{}{}".format(Prob_col, Prob_col, n_rows+1)

    logging.info("write_to_sheet() -- ID_range = {}, Species_range = {}, Prob_range = {}".
                 format(ID_range, Species_range, Prob_range))

    ss_species = worksheet.batch_get([Species_range])  # todo Fix issue where empty spreadsheet col returns error
    ss_species = [item for sublist in ss_species for item in sublist]  # Flatten list of lists
    ss_species.extend([[]] * (n_rows - len(ss_species)))  # Extend list where empty rows were cut off

    ss_prob = worksheet.batch_get([Prob_range])  # todo Fix issue where empty spreadsheet col returns error
    ss_prob = [item for sublist in ss_prob for item in sublist]  # Flatten list of lists
    ss_prob.extend([[]] * (n_rows - len(ss_prob)))

    nn_species = []
    nn_prob = []

    written_files = []
    row = []
    for i in range(n_rows):
        if results[results['tick_id'].str.match(ss_id[i][0])].empty:
            nn_species.append(ss_species[i])
            nn_prob.append(ss_prob[i])
        else:
            values = results[results['tick_id'].str.match(ss_id[i][0])]
            nn_species.append([values['prediction'].iloc[0]])
            nn_prob.append([values['prob'].iloc[0]])

            row.append(i)
            written_files.append(values['fname'].iloc[0])

    # todo add "Written" bool val and "Row" int

    written = []
    row2 = []
    for f in results['fname']:
        if f in written_files:
            written.append(True)

            idx = written_files.index(f)
            row2.append(row[idx])
        else:
            err_f.append(f)
            err_r.append("Unable to write results to sheet. Image ID was not registered")
            written.append(False)
            row2.append(np.nan)

    results['Written'] = written
    results['Row'] = row2

    worksheet.batch_update([
        {
            'range': Species_range,
            'values': nn_species},
        {
            'range': Prob_range,
            'values': nn_prob}
    ])
    print("Success")
    logging.info("write_to_sheet() -- Success")
    return results, err_f, err_r


def get_empty_preds(worksheet):
    logging.info("get_empty_preds() -- Initializing")
    col_names = worksheet.row_values(1)
    n_rows = worksheet.row_count
    logging.info("write_to_sheet() -- col_names = {}".format(col_names))
    logging.info("write_to_sheet() -- initial row count = {}".format(n_rows))

    ID_col = chr(ord('@') + col_names.index("Tick_ID") + 1)
    ID_range = "{}2:{}{}".format(ID_col, ID_col, n_rows)

    ss_id = worksheet.batch_get([ID_range])
    ss_id = [item for sublist in ss_id for item in sublist]  # Flatten list of lists

    n_rows = len(ss_id)  # Discount empty rows
    logging.info("write_to_sheet() -- row count = {}".format(n_rows))

    nn_species_col = chr(ord('@') + col_names.index("NN Species") + 1)
    nn_species_range = "{}2:{}{}".format(nn_species_col, nn_species_col, n_rows)

    nn_species = worksheet.batch_get([nn_species_range])  # todo Fix issue where empty spreadsheet col returns error
    nn_species = [item for sublist in nn_species for item in sublist]  # Flatten list of lists
    nn_species.extend([[]] * (n_rows - len(nn_species)))  # Extend list where empty rows were cut off

    logging.info("get_empty_preds() -- ID_range = {}, NN Species_range = {}".
                 format(ID_range, nn_species_range))

    fnames = []
    for idx in range(n_rows):
        if nn_species[idx] == []:
            try:
                fnames.append(ss_id[idx][0] + '.jpg')
            except IndexError:
                pass
        else:  # Prediction already exists
            pass

    return fnames


def getTicksIDs(dir):  # Get tickIDs from filenames (assumes standard name of tickid.jpg like '200515-1520-40.jpg')
    logging.info("getTicksIDs() -- Getting Tick IDs from {}".format(dir))
    fnames = os.listdir(dir)
    logging.info("getTicksIDs() -- {} files found".format(len(fnames)))
    tickID = [os.path.splitext(f)[0] for f in fnames]
    logging.info("getTicksIDs() -- Success")
    return tickID


def getSSvalues(worksheet):  # Get data from spreadsheet
    logging.info("getSSvalues() -- Initializing")
    col_names = worksheet.row_values(1)
    n_rows = worksheet.row_count
    logging.info("getSSvalues() -- col_names = {}".format(col_names))

    # Get spreadsheet ranges in the required A1 notation
    ID_col = chr(ord('@') + col_names.index("Tick_ID") + 1)
    ID_range = "{}2:{}{}".format(ID_col, ID_col, n_rows)
    ID_lst = worksheet.get(ID_range)

    n_rows = len(ID_lst)

    species_col = chr(ord('@') + col_names.index("Species_1") + 1)
    species_range = "{}2:{}{}".format(species_col, species_col, n_rows)
    species_lst = worksheet.get(species_range)
    species_lst.extend([[]] * (n_rows - len(species_lst)))  # Extend list where empty rows were cut off

    sex_col = chr(ord('@') + col_names.index("Sex_Sp1") + 1)
    sex_range = "{}2:{}{}".format(sex_col, sex_col, n_rows)
    sex_lst = worksheet.get(sex_range)
    sex_lst.extend([[]] * (n_rows - len(sex_lst)))

    lifestage_col = chr(ord('@') + col_names.index("Lifestage") + 1)
    lifestage_range = "{}2:{}{}".format(lifestage_col, lifestage_col, n_rows)
    lifestage_lst = worksheet.get(lifestage_range)
    lifestage_lst.extend([[]] * (n_rows - len(lifestage_lst)))

    fed_col = chr(ord('@') + col_names.index("Bloodfed") + 1)
    fed_range = "{}2:{}{}".format(fed_col, fed_col, n_rows)
    fed_lst = worksheet.get(fed_range)
    fed_lst.extend([[]] * (n_rows - len(fed_lst)))

    date_col = chr(ord('@') + col_names.index("Date ID") + 1)
    date_range = "{}2:{}{}".format(date_col, date_col, n_rows)
    date_lst = worksheet.get(date_range)
    date_lst.extend([[]] * (n_rows - len(date_lst)))

    logging.info("getSSvalues() -- ID_range = {}, species_range = {}, sex_range = {}, lifestage_range = {}, "
                 "fed_range = {}, date_range = {}".format(ID_range, species_range, sex_range, lifestage_range,
                                                          fed_range, date_range))

    ss_dict = {
        "Tick ID": ID_lst,
        "Species": species_lst,
        "Sex": sex_lst,
        "Lifestage": lifestage_lst,
        "Fed": fed_lst,
        "Date": date_lst
    }

    logging.info("gettSSvalues() -- Success")
    return ss_dict


def syncSS2folder(ss_dict, tickID, err_f, err_r):
    logging.info("syncSS2folder() -- {} IDs declared for syncing".format(len(tickID)))
    print("Syncing image IDs with spreadsheet data...")

    err_c = 0
    ID = []
    species = []
    sex = []
    lifestage = []
    fed = []
    date = []
    for i in tickID:
        try:
            idx = ss_dict['Tick ID'].index([i])
        except ValueError as err:  # Image is not registered on Tick_ID spreadsheet
            err_f.append(i + ".jpg")
            err_r.append("Unable to get labels. Image ID could not be found on the Tick_ID spreadsheet")
            err_c += 1
            continue

        try:
            species.append(ss_dict['Species'][idx][0])
            ID.append(i)
        except IndexError:
            err_f.append(i + ".jpg")
            err_r.append("Unable to get labels. Image has not yet been classified on the Tick_ID spreadsheet")
            err_c += 1
            continue

        try:
            sex.append(ss_dict['Sex'][idx][0])
        except IndexError:
            sex.append('None')
        try:
            lifestage.append(ss_dict['Lifestage'][idx][0])
        except IndexError:
            lifestage.append('None')
        try:
            fed.append(ss_dict['Fed'][idx][0])
        except IndexError:
            fed.append('None')
        try:
            date.append(ss_dict['Date'][idx][0])
        except IndexError:
            date.append('None')

    logging.info("syncSS2folder() -- {} match errors during syncing".format(err_c))

    data_dict = {
        "Tick ID": ID,
        "Species": species,
        "Sex": sex,
        "Lifestage": lifestage,
        "Fed": fed,
        "Date": date,
        "fname": [item + '.jpg' for item in ID]
    }

    df = pd.DataFrame.from_dict(data_dict)
    logging.info("syncSS2folder() -- Success")
    return df, err_f, err_r


def cleanData(entry_dict, df, err_f, err_r):  # Clean spreadsheet data so everything is recognized
    logging.info("cleanData() -- {} entries to clean".format(len(df)))

    # Drop any species not in entry_dict['Species'] key i.e (unknown, not a tick, etc)
    invalid_species = df[~df['Species'].str.upper().isin([s.upper() for s in entry_dict['Species']])]
    df.drop(invalid_species.index, inplace=True)
    df['Species'] = df['Species'].str.capitalize()

    logging.info("cleanData() -- {} files with invalid species names. Dropping.".format(len(invalid_species)))
    for f, s in zip(invalid_species['fname'], invalid_species['Species']):
        err_f.append(f)
        err_r.append("Unrecognized Species label '{}'".format(s))

    # Clean df['Sex'] entries
    df['Sex'] = df['Sex'].str.capitalize()
    invalid_sex = df[~df['Sex'].str.upper().isin([s.upper() for s in entry_dict['Sex']])]
    df.at[invalid_sex.index, 'Sex'] = 'Unknown'

    # Clean df['Lifestage'] entries
    df['Lifestage'] = df['Lifestage'].str.capitalize()
    invalid_lifestage = df[~df['Lifestage'].str.upper().isin([s.upper() for s in entry_dict['Lifestage']])]
    df.at[invalid_lifestage.index, 'Lifestage'] = 'Unknown'

    # Clean df['Fed'] entries
    df['Fed'] = df['Fed'].str.capitalize()
    invalid_fed = df[(~df['Fed'].str.upper().isin([s.upper() for s in entry_dict['Fed']]))]
    df.at[invalid_fed.index, 'Fed'] = 'Unknown'

    # Convert df['Date'] to datetime
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    logging.info("cleanData() -- Success")
    return df, err_f, err_r


def nameGenerator(df):  # Generates network-specific filenames like 'Dermacentor_variabilis_m_a_ta_unk_unfed_1.jpg'
    logging.info("nameGenerator() -- Generating names for {} entries".format(len(df)))

    print("Generating label-type file names...")
    df_temp = pd.DataFrame()

    df_temp['Species'] = df['Species'].str.replace(" ", "_")

    df_temp['Sex'] = df['Sex'].str.lower()
    df_temp['Sex'] = df_temp['Sex'].str.replace("unknown", "unk")

    df_temp['Lifestage'] = df['Lifestage'].str.replace("Adult", "a")
    df_temp['Lifestage'] = df_temp['Lifestage'].str.replace("Nymph", "n")
    df_temp['Lifestage'] = df_temp['Lifestage'].str.replace("Larvae", "l")
    df_temp['Lifestage'] = df_temp['Lifestage'].str.replace("Unknown", "unk")

    df_temp['Fed'] = df['Fed'].str.replace("Yes", "fed")
    df_temp['Fed'] = df_temp['Fed'].str.replace("No", "unfed")
    df_temp['Fed'] = df_temp['Fed'].str.replace("Unknown", "unk")

    df_temp["Tick ID"] = df["Tick ID"]
    df_temp["mode"] = ['ta'] * len(df_temp)
    df_temp["live"] = ['unk'] * len(df_temp)

    df['fname2'] = df_temp[['Species', 'Sex', 'Lifestage', 'mode', 'live', 'Fed', 'Tick ID']].agg('_'.join,
                                                                                                  axis=1) + '.jpg'
    logging.info("nameGenerator() -- Success")
    return df


def renameFiles(df, source, destination):  # copies and renames new files to renamed folder
    logging.info(
        "renameFiles() -- renaming {} files from source={} to destination={}".format(len(df), source, destination))
    print("Copying newly renamed images to ", destination)
    for filev1, filev2 in zip(df['fname'], df['fname2']):
        if not os.path.exists(os.path.join(destination, filev2)):
            shutil.copy2(os.path.join(source, filev1), os.path.join(destination, filev2))
    logging.info("renameFiles() -- Success")


def main(args):
    # Re-make output directory
    # try:
    #     os.makedirs(args.output)
    # except FileExistsError:
    #     shutil.rmtree(args.output)
    #     os.makedirs(args.output)

    # Check for necessary --mode arguments and raise any initialization warnings
    if args.mode == 'predict':
        if args.source is None:
            logging.error("Invalid --source argument. Choices are 'box', 'disk'")
            sys.exit(os.EX_USAGE)
        if args.path is None:
            logging.error("Invalid --path argument. Cannot be None")
            sys.exit(os.EX_USAGE)
        if args.write and args.year is None:
            logging.error("Invalid --year argument. Cannot be None when --write is True")
            sys.exit(os.EX_USAGE)
    elif args.mode == 'download':
        if args.path is None:
            logging.error("Invalid --path argument. Cannot be None")
            sys.exit(os.EX_USAGE)
        if args.destination == WMEL_raw_path or args.destination == WMEL_renamed_path:
            logging.error("Invalid --path argument. Please do not save images directly to WMEL folders with the "
                          "'download option'. Use the 'sync' option instead.")
            sys.exit(os.EX_NOPERM)
    elif args.mode == 'sync':
        if args.source is None:
            logging.error("Invalid --source argument. Choices are 'box', 'disk'")
            sys.exit(os.EX_USAGE)
        if args.path is None:
            logging.error("Invalid --path argument. Cannot be None")
            sys.exit(os.EX_USAGE)
        if args.year is None:
            logging.error("Invalid --year argument. Cannot be None")
            sys.exit(os.EX_USAGE)
        if not os.path.isdir(WMEL_raw_path):
            logging.error("Invalid WMEL Raw path. Check that drive is connected and file structure exists.")
            sys.exit(os.EX_OSFILE)
        if not os.path.isdir(WMEL_renamed_path):
            logging.error("Invalid WMEL Renamed path. Check that drive is connected and file structure exists.")
            sys.exit(os.EX_OSFILE)
    elif args.mode == 'auto':
        if args.source is None:
            logging.error("Invalid --source argument. Choices are 'box', 'disk'")
            sys.exit(os.EX_USAGE)
        if args.path is None:
            logging.error("Invalid --path argument. Cannot be None")
            sys.exit(os.EX_USAGE)
        if args.year is None:
            logging.error("Invalid --year argument. Cannot be None")
            sys.exit(os.EX_USAGE)
    else:
        pass

    # Initialize error files lists
    err_f = []
    err_r = []

    # Primary run conditions
    if args.mode == 'predict':
        if args.write:
            logging.warning("If existing predictions exist, they will be overwritten")

        if args.source == 'box':
            ftp = ftp_login()
            ftp = ftp_navigate(ftp, args.path)

            box_transfer = os.path.join(args.output, "box-transfer")
            try:
                os.makedirs(box_transfer)
            except FileExistsError:
                shutil.rmtree(box_transfer)
                os.makedirs(box_transfer)

            box_download(ftp, box_transfer, args.check_path)

            model = loadModel(args.model_dir)  # Load model and evaluate images
            df_results, err_f, err_r = NNPredict(model, box_transfer, os.listdir(box_transfer), classes, err_f, err_r)

        elif args.source == 'disk':
            if args.disk_transfer:
                disk_transfer = os.path.join(args.output, "disk-transfer")
                try:
                    os.makedirs(disk_transfer)
                except FileExistsError:
                    shutil.rmtree(disk_transfer)
                    os.makedirs(disk_transfer)

                for f in os.listdir(args.path):
                    shutil.copy2(os.path.join(args.path, f), os.path.join(disk_transfer, f))

            model = loadModel(args.model_dir)
            df_results, err_f, err_r = NNPredict(model, args.path, os.listdir(args.path),
                                                 classes, err_f, err_r)

        if args.write:
            worksheet = initSheet(service_acc, ss, args.year)
            df_results, err_f, err_r = write_to_sheet(worksheet, df_results, err_f, err_r)
            # todo Return comparison of predictions and labels if possible
            # todo Add images not registered on spreadsheet to error files

        df_results.sort_values(by='fname')
        df_results.to_csv(os.path.join(args.output, "pred-results.csv"))

        err_dict = {
            "fname": err_f,
            "Reason": err_r
        }
        df_err = pd.DataFrame.from_dict(err_dict)
        df_err.to_csv(os.path.join(args.output, "error.csv"))

        # todo Return results, comparison (if write), log files, and error files

    elif args.mode == 'download':
        ftp = ftp_login()
        ftp = ftp_navigate(ftp, args.path)

        if args.destination is None:
            box_transfer = os.path.join(args.output, "box-transfer")
            try:
                os.makedirs(box_transfer)
            except FileExistsError:
                shutil.rmtree(box_transfer)
                os.makedirs(box_transfer)
            new_files = box_download(ftp, box_transfer, args.check_path)
        else:
            new_files = box_download(ftp, args.destination, args.check_dir)

    elif args.mode == 'sync':
        if args.source == 'box':
            ftp = ftp_login()
            ftp = ftp_navigate(ftp, args.path)

            new_files = box_download(ftp, WMEL_raw_path, WMEL_raw_path)
        elif args.source == 'disk':
            new_files = []
            for f in os.listdir(args.path):
                if not os.path.exists(os.path.join(WMEL_raw_path, f)):
                    new_files.append(f)
                    shutil.copy2(os.path.join(args.path, f), os.path.join(WMEL_raw_path, f))

        matchExtensions(WMEL_raw_path, '.jpg')
        tick_id = getTicksIDs(WMEL_raw_path)

        worksheet = initSheet(service_acc, ss, args.year)
        ss_dict = getSSvalues(worksheet)
        df_files, err_f, err_r = syncSS2folder(ss_dict, tick_id, err_f, err_r)
        df_files, err_f, err_r = cleanData(entry_dict, df_files, err_f, err_r)

        df_files = nameGenerator(df_files)
        renameFiles(df_files, WMEL_raw_path, WMEL_renamed_path)

        # todo Raise error and exit if --source='disk' or 'box' but dir does not contain tick-id type images

        df_files.sort_values(by='fname')
        df_files.to_csv(os.path.join(args.output, "sync-files.csv"))

        df_new_files = df_files.loc[df_files['fname'].isin(new_files)]
        df_new_files.sort_values(by='fname')
        df_new_files.to_csv(os.path.join(args.output, "new-sync-files.csv"))

        err_dict = {
            "fname": err_f,
            "Reason": err_r
        }
        df_err = pd.DataFrame.from_dict(err_dict)
        df_err.to_csv(os.path.join(args.output, "error.csv"))

    elif args.mode == 'auto':
        logging.warning("If existing predictions exist, they will be overwritten")
        worksheet = initSheet(service_acc, ss, args.year)
        fnames = get_empty_preds(worksheet)

        if args.source == 'box':
            ftp = ftp_login()
            ftp = ftp_navigate(ftp, args.path)

            box_transfer = os.path.join(args.output, "box-transfer")
            try:
                os.makedirs(box_transfer)
            except FileExistsError:
                shutil.rmtree(box_transfer)
                os.makedirs(box_transfer)

            box_files = ftp.nlst()  # get filenames within the directory
            box_files.remove('.')
            box_files.remove('..')

            new_files = []
            for f in box_files:
                if f in fnames:
                    new_files.append(f)

            if new_files:
                for f in tqdm(new_files):
                    filename = os.path.join(box_transfer, f)

                    img = open(filename, 'wb')
                    ftp.retrbinary('RETR ' + f, img.write)
                    img.close()

            ftp.quit()  # close connection

            model = loadModel(args.model_dir)
            df_results, err_f, err_r = NNPredict(model, box_transfer, os.listdir(box_transfer), classes, err_f, err_r)

        elif args.source == 'disk':
            new_files = []
            for f in os.listdir(args.path):
                if f in fnames:
                    new_files.append(os.path.join(args.path, f))

            if args.disk_transfer:
                disk_transfer = os.path.join(args.output, "disk-transfer")
                try:
                    os.makedirs(disk_transfer)
                except FileExistsError:
                    shutil.rmtree(disk_transfer)
                    os.makedirs(disk_transfer)

                for f in new_files:
                    shutil.copy2(f, os.path.join(disk_transfer, os.path.basename(f)))

            model = loadModel(args.model_dir)
            df_results, err_f, err_r = NNPredict(model, args.path, new_files, classes, err_f, err_r)

        worksheet = initSheet(service_acc, ss, args.year)
        df_results, err_f, err_r = write_to_sheet(worksheet, df_results, err_f, err_r)

        df_results.sort_values(by='fname')
        df_results.to_csv(os.path.join(args.output, "pred-results.csv"))

        err_dict = {
            "fname": err_f,
            "Reason": err_r
        }
        df_err = pd.DataFrame.from_dict(err_dict)
        df_err.to_csv(os.path.join(args.output, "error.csv"))

        # todo Return comparison of predictions and labels if possible
        # todo Return results, comparison (if write), log files, and error files


if __name__ == '__main__':
    args = read_args()
    main(args)
