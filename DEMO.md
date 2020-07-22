# TickNet Demo

## 1) Setup

### Clone TickNet Repository for Github
```
cd /Users/Lenni/Desktop/MCEVBD-Example
git clone https://github.com/lennijusten/TickNet.git
```
### Image_ID Spreadsheet
I have created a copy of the Image_ID spreadsheet and added two columns: `NN Species` and `NN Prob`. In order to allow the script to access the spreadsheet, I need to share it with a google service acount email: `mcevbd@ticknet.iam.gserviceaccount.com`. The script will call the associated JSON file `service_account.json` so we need to move the service account file into the `/TickNet` repository.

**Note:** There is a wierd bug in the Google Sheets API that requires there to be at least one entry in the column in order for python to be able to pull and push values. I have added two place-holders (these can be removed later or overwritten).

### Model 
The model directory contains the network structure and the saved weights (large file!). Let's move those into the `/TickNet` repository as well.

### Activate conda environment with the required packages (see installation)
```
conda activate venv2
```

## 2) Running TickApp_run.py
First, let's change our current working directory (cwd) to the `TickNet` repository.
```
cd /TickNet
```

### Let's download new images form the `/tick-id-images` folder in Box that have not yet been transferred to the WMEL Drive
```
python TickApp_run.py --mode=download --source=box --path="tick-id-images" --check_path="/Volumes/WMEL/Tick Images/NN Images/Raw (if exists)/Tick ID images 2020"
```
Notice the default destination of the images. If you want to download them somewhere other than `/TickNet/output/box-transfer` use the `--destination=` argument. The `box-files.csv` returns the filenames of the images that were downloaded.

### Let's make a prediction on those very same images
```
python TickApp_run.py --mode=predict --source=disk --path="/Users/Lenni/Desktop/MCEVBD-Example/TickNet/output/box-transfer" --write --year=2020
```
We can check the spreadsheet to confirm the results were written and then look at the output. 

### What about the `--auto` mode?
Notice that we just ran the script twice: once to download new images from Box and once to make and push the predictions to Google. The `auto` mode makes this even easier by checking all the images in a specified folder against the spreadsheet and loading/predicting/writing only images that are not yet registered on the spreadsheet. 
```
python TickApp_run.py --mode=auto --source=box --path="tick-id-images" --year=2020
```

### Last but not least, the `--sync` mode
```
python TickApp_run.py --mode=sync --source=disk --path="/Users/Lenni/Desktop/MCEVBD-Example/TickNet/output/box-transfer" --year=2020

```
