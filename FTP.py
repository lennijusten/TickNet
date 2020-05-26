# Box FTP connection
# Author: Lennart Justen
# Last revision: 5/22/20

# Description: This script accesses my Box account with an FTP TLS (SSL) connection.
# It then navigates to a folder with the ftp.cwd() command, retrieves the files, and
# writes them to a local directory specified in the local_filename variable.

# Documentation
# https://docs.python.org/3/library/ftplib.html
# https://stackoverflow.com/questions/5230966/python-ftp-download-all-files-in-directory
# https://support.box.com/hc/en-us/articles/360043697414-Using-Box-with-FTP-or-FTPS?flash_digest=ac78d917845fe263bbb6b615940e10421a270626

from ftplib import FTP_TLS
import os.path

local_dir_path = 'path/'

ftp = FTP_TLS('ftp.box.com')
print("Welcome: ", ftp.getwelcome())
ftp.login(user='username', passwd='password')

main_dir = ftp.retrlines('LIST')
ftp.pwd()
ftp.cwd('WMEL-Tick Images/Submitted Images')
print(ftp.pwd())

filenames = ftp.nlst()  # get filenames within the directory
filenames.remove('.')
filenames.remove('..')
print(filenames)

new_count = 0
exist_count = 0
for filename in filenames:
    local_filename = os.path.join(local_dir_path, filename)
    if not os.path.exists(local_filename):
        new_count = new_count + 1
        file = open(local_filename, 'wb')
        ftp.retrbinary('RETR ' + filename, file.write)

        file.close()
    else:
        exist_count = exist_count + 1

ftp.quit()  # close connection
print(new_count, " new images were downloaded")
print(exist_count, " images already existed in local directory:", local_dir_path)
