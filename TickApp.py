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

# todo Send relevant people service_account json file and add instructions on where to move it
# todo Check WMEL file paths
# todo Update entry_dict, service account, spreadsheet, and classes

classes = ['Amblyomma americanum', 'Dermacentor variabilis', 'Ixodes scapularis']
ss = 'Copy of Image_ID_July2'
service_acc = '/Users/Lenni/Documents/PycharmProjects/MCEVBD Fellowship/service_account.json'

entry_dict = {
    "Species": ['Dermacentor variabilis', 'Dermacentor andersoni',
                'Ixodes scapularis',
                'Amblyomma americanum', 'Amblyomma maculatum'],
    "Sex": ['M', 'F'],
    "Lifestage": ['Adult', 'Nymph', 'Larvae'],
    "Fed": ['yes', 'no']
}


def read_args():    # Get user arguments
    now = datetime.datetime.now()

    parser = argparse.ArgumentParser(description="Tick App prediction node")

    parser.add_argument("--source",
                        type=str,
                        choices=['box', 'disk'],
                        default=None,
                        help="image source ['box', 'disk']")

    parser.add_argument("--path",
                        type=str,
                        default=None,
                        help="path to image/s")

    parser.add_argument("--mode",
                        type=str,
                        choices=['predict', 'sync', 'auto'],
                        default='auto',
                        help="run mode (default='auto')")

    parser.add_argument("--model_dir",
                        type=str,
                        default=None,
                        help="path to model dir with .json and .h5 files")

    parser.add_argument("--output",
                        type=str,
                        default='output',
                        help="output directory")

    parser.add_argument("--write",
                        action="store_true",
                        help="write results to spreadsheet")

    parser.add_argument("--year",
                        type=int,
                        default=now.year,
                        help="singular year from which to sync data")

    parser.add_argument("--rename",
                        action="store_true",
                        help="rename files in standard format")

    parser.add_argument("--WMEL_drive",
                        action="store_true",
                        help="sync data to existing WMEL external hard drive file structure")

    parser.add_argument("--keep",
                        action="store_true",
                        help="keep existing output images")

    args = parser.parse_args()
    return args


args = read_args()


def ftp_login():    # Login to box server
    username = input("BOX username: ")  # Get user input
    password = getpass.getpass(prompt="BOX password:")  # Get user password (secure)

    ftp = FTP_TLS('ftp.box.com')
    print("Accessing box.com with login credentials...")
    ftp.login(user=username, passwd=password)
    print("Success")
    return ftp


def ftp_navigate(ftp, path): # navigate to box path
    ftp.cwd(path)
    return ftp


def box_download(ftp, check_dir, destination):  # Download box files with ftp connection (slow)

    box_files = ftp.nlst()  # get filenames within the directory
    box_files.remove('.')
    box_files.remove('..')

    existing_files = []
    new_files = []
    for f in box_files:     # Check if files exist in check_dir
        if glob.glob(os.path.join(check_dir, os.path.splitext(f)[0] + ".*")):
            existing_files.append(f)
        else:
            new_files.append(f)

    print("{} new images, {} existing images on the local disk, {} total images in box dir ".
          format(len(new_files), len(existing_files), len(box_files)))

    if new_files:   # Retrieve new images and write to disk
        print("Retrieving new images...")
        for f in tqdm(new_files):
            local_filename = os.path.join(destination, f)

            img = open(local_filename, 'wb')
            ftp.retrbinary('RETR ' + f, img.write)
            img.close()

        ftp.quit()  # close connection


def get_input(path):    # Load image
    img = Image.open(path)
    return img


def im_crop(image):     # Crop images into a square along their shortest side
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


def matchExtensions(dir, extension):    # standardize image extensions (.JPG, .jpeg, .png, .PNG, etc)
    for filename in os.listdir(dir):
        infilename = os.path.join(dir, filename)
        if not os.path.isfile(infilename): continue
        base = os.path.splitext(infilename)[0]
        os.rename(infilename, base + extension)


def loadModel(path):    # Load neural network with saved weights and architecture
    json_path = glob.glob(os.path.join(path, "*.json"))     # look for any file with .json extension
    weights_path = glob.glob(os.path.join(path, "*.h5"))    # look for any file with .h5 extension

    json_file = open(json_path[0], 'r')     # load first model instance
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path[0])  # load first weights instance

    print("Model loaded from disk.")
    return loaded_model


def NNPredict(model, path, classes, err_f, err_r):  # Make predictions
    print("Evaluating images...")
    input_shape = model.layers[0].input_shape   # get required image input size (model specific)

    fname = []
    tick_id = []
    images = []
    for f in tqdm(os.listdir(path)):
        try:
            img = get_input(os.path.join(path, f))
            img_cropped = im_crop(img)
            img_resized = img_cropped.resize((input_shape[1], input_shape[2]))
            pixels = np.asarray(img_resized)    # convert image to array
            pixels = pixels.astype('float32')
            pixels /= 255.0     # rescale image from 0-255 to 0-1 (best practice)
            input = np.expand_dims(pixels, axis=0)  # adds batch dimension
            images.append(input)

            fname.append(f)
            tick_id.append(os.path.splitext(f)[0])
        except:
            err_f.append(f)
            err_r.append("Image could not be opened")
            pass

    # stack up images list to pass for prediction
    images = np.vstack(images)

    results = model.predict(images, batch_size=32)

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
    return df, err_f, err_r


def initSheet(service_acc, ss, year):   # Connect to google sheets api with service account
    print("Connecting to Google Sheets {} with login credentials...".format(ss))
    gc = gspread.service_account(filename=service_acc)
    sh = gc.open(ss)
    print("Opening {} sheet...".format(str(year)))
    worksheet = sh.worksheet(str(year))
    print("Success")
    return worksheet


def write_to_sheet(worksheet, results): # Write results to google sheets with gspread batch api
    print("Writing results to worksheet...")
    col_names = worksheet.row_values(1)
    n_rows = worksheet.row_count

    ID_col = chr(ord('@') + col_names.index("Tick_ID") + 1)
    ID_range = "{}2:{}{}".format(ID_col, ID_col, n_rows)

    ss_id = worksheet.batch_get([ID_range])
    ss_id = [item for sublist in ss_id for item in sublist]  # Flatten list of lists

    n_rows = len(ss_id)  # Discount empty rows

    Species_col = chr(ord('@') + col_names.index("NN Species") + 1)
    Species_range = "{}2:{}{}".format(Species_col, Species_col, n_rows)

    Prob_col = chr(ord('@') + col_names.index("NN Prob") + 1)
    Prob_range = "{}2:{}{}".format(Prob_col, Prob_col, n_rows)

    ss_species = worksheet.batch_get([Species_range])
    ss_species = [item for sublist in ss_species for item in sublist]  # Flatten list of lists
    ss_species.extend([[]] * (n_rows - len(ss_species)))  # Extend list where empty rows were cut off

    ss_prob = worksheet.batch_get([Prob_range])
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

    pd.set_option('mode.chained_assignment', None)
    results['Written'] = results['fname'].isin(written_files)
    results['Row'] = np.nan
    results['Row'].loc[results['Written'] == True] = row

    worksheet.batch_update([{
        'range': Species_range,
        'values': nn_species},
        {
            'range': Prob_range,
            'values': nn_prob
        }])
    print("Success")

    return results


def getTicksIDs(dir):   # Get tickIDs from filenames (assumes standard name of tickid.jpg like '200515-1520-40.jpg')
    fnames = os.listdir(dir)
    tickID = [os.path.splitext(f)[0] for f in fnames]
    return tickID


def getSSvalues(worksheet): # Get data from spreadsheet
    col_names = worksheet.row_values(1)
    n_rows = worksheet.row_count

    # Get spreadsheet ranges in the required A1 notation

    ID_col = chr(ord('@') + col_names.index("Tick_ID") + 1)
    ID_range = "{}2:{}{}".format(ID_col, ID_col, n_rows)

    species_col = chr(ord('@') + col_names.index("Species_1") + 1)
    species_range = "{}2:{}{}".format(species_col, species_col, n_rows)

    sex_col = chr(ord('@') + col_names.index("Sex_Sp1") + 1)
    sex_range = "{}2:{}{}".format(sex_col, sex_col, n_rows)

    lifestage_col = chr(ord('@') + col_names.index("Lifestage") + 1)
    lifestage_range = "{}2:{}{}".format(lifestage_col, lifestage_col, n_rows)

    fed_col = chr(ord('@') + col_names.index("Bloodfed") + 1)
    fed_range = "{}2:{}{}".format(fed_col, fed_col, n_rows)

    date_col = chr(ord('@') + col_names.index("Date ID") + 1)
    date_range = "{}2:{}{}".format(date_col, date_col, n_rows)

    ID_lst = [item for sublist in worksheet.get(ID_range) for item in (sublist or [''])]
    species_lst = [item for sublist in worksheet.get(species_range) for item in (sublist or [''])]
    sex_lst = [item for sublist in worksheet.get(sex_range) for item in (sublist or [''])]
    lifestage_lst = [item for sublist in worksheet.get(lifestage_range) for item in (sublist or [''])]
    fed_lst = [item for sublist in worksheet.get(fed_range) for item in (sublist or [''])]
    date_lst = [item for sublist in worksheet.get(date_range) for item in (sublist or [''])]

    ss_dict = {
        "Tick ID": ID_lst,
        "Species": species_lst,
        "Sex": sex_lst,
        "Lifestage": lifestage_lst,
        "Fed": fed_lst,
        "Date": date_lst
    }

    return ss_dict


def syncSS2folder(ss_dict, tickID, err_f, err_r):
    print("Syncing image IDs with spreadsheet data...")

    ID = []
    species = []
    sex = []
    lifestage = []
    fed = []
    date = []
    for i in tickID:
        try:
            idx = ss_dict['Tick ID'].index(i)

            species.append(ss_dict['Species'][idx])
            sex.append(ss_dict['Sex'][idx])
            lifestage.append(ss_dict['Lifestage'][idx])
            fed.append(ss_dict['Fed'][idx])
            date.append(ss_dict['Date'][idx])

            ID.append(i)
        except ValueError as err:  # Image is not registered on Tick_ID spreadsheet
            err_f.append(i + ".jpg")
            err_r.append("Unable to get labels. Image ID could not be found on the Tick_ID spreadsheet")
        except IndexError:  # Image is not yet been identified on Tick_ID spreadsheet
            err_f.append(i + ".jpg")
            err_r.append("Unable to get labels. Image has not yet been classified on the Tick_ID spreadsheet")
            break

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
    return df, err_f, err_r


def cleanData(entry_dict, df, err_f, err_r):    # Clean spreadsheet data so everything is recognized

    # Drop any species not in entry_dict['Species'] key i.e (unknown, not a tick, etc)
    invalid_species = df[~df['Species'].str.upper().isin([s.upper() for s in entry_dict['Species']])]
    df.drop(invalid_species.index, inplace=True)
    df['Species'] = df['Species'].str.capitalize()

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
    df.at[invalid_lifestage.index, 'Fed'] = 'Unknown'

    # Convert df['Date'] to datetime
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    return df, err_f, err_r


def nameGenerator(df):      # Generates network-specific filenames like 'Dermacentor_variabilis_m_a_ta_unk_unfed_1.jpg'

    print("Generating label-type file names...")
    df_temp = pd.DataFrame()

    df_temp['Species'] = df['Species'].str.replace(" ", "_")

    df_temp['Sex'] = df['Sex'].str.lower()
    df_temp['Sex'] = df_temp['Sex'].str.replace("unknown", "unk")  # todo Fix

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
    return df


def renameFiles(df, source, destination):  # copies and renames new files to renamed folder
    print("Copying newly renamed images to ", destination)
    for filev1, filev2 in zip(df['fname'], df['fname2']):
        if not os.path.exists(os.path.join(destination, filev2)):
            shutil.copy2(os.path.join(source, filev1), os.path.join(destination, filev2))


def main(args):
    # Initialize error logs
    df_err = pd.DataFrame()
    err_f = []
    err_r = []

    # Check for existing output folder
    if args.keep and os.path.exists(args.output):
        pass
    elif args.keep and not os.path.exists(args.output):
        print("Warning: -keep was specified but the --output directory does not exist. Making new dir...")
        os.makedirs(args.output)
    else:
        try:  # If args.keep = False, delete output dir tree and re-initialize
            os.makedirs(args.output)
        except FileExistsError:
            shutil.rmtree(args.output)
            os.makedirs(args.output)

    if args.source == 'box':
        ftp = ftp_login()
        ftp = ftp_navigate(ftp, args.path)

        if args.mode == 'predict':

            try:  # Creates transfer folder to store images on disk
                os.makedirs(os.path.join(args.output, "box transfer"))
            except FileExistsError:
                pass

            if args.keep:  # Check contents of output folder and download only new images
                box_download(ftp, os.path.join(args.output, "box transfer"), os.path.join(args.output, "box transfer"))
            else:
                box_download(ftp, args.output, os.path.join(args.output, "box transfer"))

            model = loadModel(args.model_dir)  # Load model and evaluate images
            df_results, err_f, err_r = NNPredict(model, os.path.join(args.output, "box transfer"), classes, err_f,
                                                 err_r)

            if args.write:  # Write results to google sheets
                worksheet = initSheet(service_acc, ss, args.year)
                df_results = write_to_sheet(worksheet, df_results)

            # Write logs to output folder
            df_err['fname'], df_err['Reason'] = err_f, err_r
            df_results, df_err = df_results.sort_values(by="fname"), df_err.sort_values(by="fname")
            df_err.to_csv(os.path.join(args.output, "error-files.csv"))
            df_results.to_csv(os.path.join(args.output, "prediction-results.csv"))

        elif args.mode == 'sync':

            if args.WMEL_drive:
                raw_path = '/Volumes/WMEL/Tick Images/NN Images/Raw (if exists)/Tick ID images {}'
                raw_path = raw_path.format(str(args.year))

                box_download(ftp, raw_path, raw_path)
                tick_id = getTicksIDs(raw_path)
                worksheet = initSheet(service_acc, ss, args.year)

                ss_dict = getSSvalues(worksheet)
                df_files, err_f, err_r = syncSS2folder(ss_dict, tick_id, err_f,
                                                       err_r)

                df_files, err_f, err_r = cleanData(entry_dict, df_files, err_f, err_r)

                if args.rename:
                    renamed_path = '/Volumes/WMEL/Tick Images/NN Images/Renamed (all)/Tick ID images {}'
                    renamed_path = renamed_path.format(str(args.year))

                    df_files = nameGenerator(df_files)
                    renameFiles(df_files, raw_path, renamed_path)
                else:
                    pass

                df_err['fname'], df_err['Reason'] = err_f, err_r
                df_files, df_err = df_files.sort_values(by="fname"), df_err.sort_values(by="fname")
                df_err.to_csv(os.path.join(args.output, "error-files.csv"))
                df_files.to_csv(os.path.join(args.output, "files.csv"))
            else:
                try:
                    os.makedirs(os.path.join(args.output, "Raw"))
                except FileExistsError:
                    pass

                try:
                    os.makedirs(os.path.join(args.output, "Raw", "Tick ID images {}".format(args.year)))
                except FileExistsError:
                    pass

                raw_path = os.path.join(args.output, "Raw/Tick ID images {}".format(args.year))

                box_download(ftp, raw_path, raw_path)
                tick_id = getTicksIDs(raw_path)
                worksheet = initSheet(service_acc, ss, args.year)

                ss_dict = getSSvalues(worksheet)
                df_files, err_f, err_r = syncSS2folder(ss_dict, tick_id, err_f, err_r)

                df_files = cleanData(entry_dict, df_files)

                if args.rename:
                    try:
                        os.makedirs(os.path.join(args.output, "Renamed"))
                    except FileExistsError:
                        pass

                    try:
                        os.makedirs(os.path.join(args.output, "Renamed", "Tick ID images {}".format(args.year)))
                    except FileExistsError:
                        pass

                    renamed_path = os.path.join(args.output, "Renamed", "Tick ID images {}".format(args.year))

                    df_files = nameGenerator(df_files)
                    renameFiles(df_files, raw_path, renamed_path)
                else:
                    pass

                df_err['fname'], df_err['Reason'] = err_f, err_r
                df_files, df_err = df_files.sort_values(by="fname"), df_err.sort_values(by="fname")
                df_err.to_csv(os.path.join(args.output, "error-files.csv"))
                df_files.to_csv(os.path.join(args.output, "files.csv"))
        elif args.mode == 'auto':
            try:
                os.makedirs(args.output)
            except FileExistsError:
                shutil.rmtree(args.output)
                os.makedirs(args.output)
                os.makedirs(os.path.join(args.output, "box transfer"))

            raw_path = '/Volumes/WMEL/Tick Images/NN Images/Raw (if exists)/Tick ID images {}'  # todo Check path
            raw_path = raw_path.format(str(args.year))

            # Save images to transfer directory and then copy to WMEL drive
            box_download(ftp, raw_path, os.path.join(args.output, "box transfer"))
            for f in os.listdir(os.path.join(args.output, "box transfer")):
                shutil.copy2(os.path.join(args.output, "box transfer", f), os.path.join(raw_path, f))

            tick_id = getTicksIDs(raw_path)
            worksheet = initSheet(service_acc, ss, args.year)

            ss_dict = getSSvalues(worksheet)
            df_files, err_f, err_r = syncSS2folder(ss_dict, tick_id, err_f, err_r)

            df_files, err_f, err_r = cleanData(entry_dict, df_files, err_f, err_r)

            renamed_path = '/Volumes/WMEL/Tick Images/NN Images/Renamed (all)/Tick ID images {}'    # todo Check path
            renamed_path = renamed_path.format(str(args.year))

            df_files = nameGenerator(df_files)
            renameFiles(df_files, raw_path, renamed_path)

            model = loadModel(args.model_dir)
            df_results, err_f, err_r = NNPredict(model, os.path.join(args.output, "box transfer"), classes, err_f,
                                                 err_r)

            worksheet = initSheet(service_acc, ss, args.year)
            write_to_sheet(worksheet, df_results)

            df_err['fname'], df_err['Reason'] = err_f, err_r
            df_results, df_files, df_err = df_results.sort_values(by='fname'), \
                                           df_files.sort_values(by='fname'), df_err.sort_values(by="fname")
            df_results.to_csv(os.path.join(args.output, "prediction-results.csv"))
            df_err.to_csv(os.path.join(args.output, "error-files.csv"))
            df_files.to_csv(os.path.join(args.output, "files.csv"))
        else:
            pass
    elif args.source == 'disk':

        model = loadModel(args.model_dir)
        df_results, err_f, err_r = NNPredict(model, args.path, classes, err_f, err_r)

        if args.write:
            worksheet = initSheet(service_acc, ss, args.year)
            df_results = write_to_sheet(worksheet, df_results)

        df_err['fname'], df_err['Reason'] = err_f, err_r
        df_results, df_err = df_results.sort_values(by="fname"), df_err.sort_values(by="fname")
        df_err.to_csv(os.path.join(args.output, "error-files.csv"))
        df_results.to_csv(os.path.join(args.output, "prediction-results.csv"))

    else:
        pass


if __name__ == '__main__':
    args = read_args()
    main(args)
