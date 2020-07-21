# TickNet

## 1. Installation
The following lines can be copied into your console (called Terminal on mac). 

### Anaconda (recommended)
```bash
conda create --name venv python=3.6
conda activate venv
conda install tensorflow pillow tqdm pandas
conda install -c conda-forge gspread
```

### Virtualenv
```bash
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Model
The model (.json file) and the model weights (.h5 files) are stored in the **Model/** directory. The model may be updated from time to time with the newest and hopefully best version. 

## 3. WMEL External Hard-drive
The complete tick-image database is stored on the WMEL external hard-drive. This contains all the images used for training and is designed to be continuously updated as more images become available. 

The WMEL drive contains two main image directories: 
```
/Volumes/WMEL/Tick Images/NN Images/Raw (if exists)
/Volumes/WMEL/Tick Images/NN Images/Renamed (all)
```

The Raw image folder contains the raw images from some of data sources: Tick App (ta), WMEL Lab (MCEVBD). The raw images are not augmented at all and contain the original file names of the images. The Renamed folder contains the renamed image files from all the data sources: Tick App, WMEL Lab (MCEVBD), TickReport (tr), Lab Images (DSLR & mb). The network training script pulls directly from this folder so it is essential that this structure remains consistent! 

The renamed images are all named accoring to a standard naming system which uses the following structure:
```
Genus_species_sex_lifestage_source_alive_fed_#.jpg
```
For example: `Dermacentor_variabilis_m_a_ta_unk_unfed_1.jpg`

The labels have the following categories:  
**Genus:** any  
**species:** any  
**sex:** male (m), female (f), unkown (unk)  
**lifestage:** adult (a), nymph (n), larvae (l), unkown (unk)  
**source:** Tick App (ta), WMEL Lab (MCEVBD), TickReport (tr), Images I took during 2020 in the WMEL Lab (DSLR or mb)   
**alive:** dead (dead), alive (live), unkown (unk)   
**fed:** fed (fed), unfed (unfed), unkown (unk)   


## 4. Usage (TickApp)
The `TickApp_run.py` script has several functions designed to classify tick images, send results to the Image_ID spreadsheet, and tranfer images between Box, your local disk, and the WMEL external hard-drive. Below are some plain language descriptions of the functionality as well as some common usage examples. For a more traditional documentation, please see the `documentation.pdf` in the repository. 

**A note on Box:** Any mode that accesses Box for file transfers will require the user to log-in. The script will prompt you for your username and for your password. Neither the username nor password are stored. Additionally, the password entry is secure so it will not be echoed (reprinted) to the console. 

**Running the script:** The `TickApp_run.py` is a python script that can be run from your machines console (Terminal). Before running, you need to make sure that your environment is activated (See installation) and that your current directory is set to the TickNet repository. You can use the `cd <path_to_repository>` command to move between directories. Two other helpful commands for navigation are the `pwd` command which shows you what your current directory is and the `ls` command which lists the contents of you current directory.

### Auto 
The `auto` mode is designed to make predictions on images that do not have an existing prediction entries on the Image_ID spreadsheet. The most likely application of this mode is in conjunction with the "tick-id-images" directory where new images are stored. In the `auto` mode, the results are automatically written to the spreadsheet specified with the `--year=` argument. Below are some usage examples:

To make predictions on the newest batch of images and send the results to the 2020 Image_ID spreadsheet:
```
python TickApp_run.py --mode=auto --source=box --path="tick-id-images" --year=2020
```
The `auto` mode has an additional feature specified by the `--disk_transfer` argument that moves a copy of the predicted images to the output directory:
```
python TickApp_run.py --mode=auto --source=box --path="tick-id-images" --year=2020 --disk_transfer
```
**Note:** Unlike the `predict` mode, the `auto` mode will not overwrite old predictions.


### Predict
The `predict` mode is used to make predictions on a directory of images. With the `--source=` argument, you can specify whether the you want to predict a folder in Box or a folder on your local disk. The `--path=` argument specifies the path to the directory. With the `--write` argument, you can choose to send the results to the "Image_ID" spreadsheet, however you must then also specify the `--year=` argument which makes sure the results are sent to the right sheet. Below are some usage examples. 

To return a csv file with the prediction results from images in a local folder:
```
python TickApp_run.py --mode=predict --source=disk --path="/Users/JaneDoe/Desktop/TickImages"
```
To return a csv file with prediction results from images in a box folder and send the results to the 2020 Image_ID Spreadsheet:
```
python TickApp_run.py --mode=predict --source=box --path="tick-id-images" --write --year=2020
```
Both of these options will return a `prediction-results.csv` file as well as an `error.csv` file which contains any images that could not be opened or were not found on the spreadsheet. 

### Sync (requires WMEL external hard-drive to be connected)
The `sync` mode syncs the WMEL drive data with the Image_ID spreadsheet. With the `--source=` and `--path=` arguments, any new images will be moved into the WMEL database, matched with labels from the Image_ID spreadsheet which must again be specified with the `--year=` argument, and renamed using the standard naming system. Below are some usage examples.

To sync images from the 2020 images box folder to the WMEL drive:
```
python TickApp_run.py --mode=sync --source=box --path="Tick App Pictures/Education file/Tick ID images 2020" --year=2020
```
**Note:** Be careful to ensure that the folder contains images from the same year as the `--year` argument otherwise the WMEL file structure will become disordered and the images labels won't be found on the spreadsheet.

To sync 2019 images from your local disk to the WMEL drive:
```
python TickApp_run.py --mode=sync --source=disk --path="Users/John/Desktop/2019-tickapp-images" --year=2019
```
The sync mode will return a CSV file of ALL the images in the respective WMEL drive dir, a CSV file of the newly synced data, and an error CSV containing any images that were not registered on the spreadsheet. 

### Download
The `download` mode downloads images from box to a specified folder on your disk. The mode has a useful argument called the `--check_path` which can be set so that the script will only download images that are NOT found in the `--check_path` dir. The download mode uses FTP file transfer which can very (very very) slow so for large file transfers I recommend using the box website to just download images. Below are some usage examples:

To download a whole box folder onto your local disk (not recommended for large folders >50 images)
```
python TickApp_run.py --mode=download --path="tick-id-images" --destination="Users/Lenni/Downloads/Images"
```
To download the newest images from box to a folder that already contains many of the images in the box folder:
```
python TickApp_run.py --mode=download --path="tick-id-images" --destination="Users/Lenni/Downloads/Images" --check_path="Users/Lenni/Downloads/Images"
```
To download the newest images that have not been synced to the WMEL drive to a folder on your disk:
```
python TickApp_run.py --mode=download --path="tick-id-images" --destination="Users/Lenni/Downloads/Images" --check_path="/Volumes/WMEL/Tick Images/NN Images/Raw (if exists)/Tick ID images 2020"
```
To make a prediction on those images and then write them to the spreadsheet (review):
```
python TickApp_run.py --mode=predict --source=disk --path="Users/Lenni/Downloads/Images" --write --year=2020
```
**Note:** You should not use the download mode to download images directly into the WMEL drive. This risks breaking certain existing file structures. Use the `sync` mode instead. 

## Contributing and Issue Reporting
The contribute or report issues please use Github's `Pull Requests` and `Issues` features. If reporting an issue/bug, please attach the console output so I can better address the issue. What is also helpful is recreating the error but with the `--logging` argument which will produce a more detailed console log. 

I can be contacted at ljusten@wisc.edu
