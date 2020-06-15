import gspread
ss = "Copy of Image_ID"

species = 'Ixodes scapularis'
species_prob = '93.63'
sex = 'F'
# bloodfed = 'yes'
prob = '80.34'

try:
    gc = gspread.service_account(filename='/Users/Lenni/Documents/PycharmProjects/MCEVBD Fellowship/service_account.json')
except FileNotFoundError as err:
    print("Unrecognized service account file: {}".format(err))

try:
    sh = gc.open(ss)
except gspread.exceptions.SpreadsheetNotFound as err:
    print("Error: Spreadsheet '{}' not found".format(ss))

worksheet = sh.sheet1
column_headers = worksheet.row_values(1)
print(column_headers)

ID = '200203-1815-21'
rows = worksheet.findall(ID, in_column=worksheet.find("Tick_ID").col)
if len(rows) == 0:
    print("Tick ID '{}' was not found".format(ID))
elif len(rows) > 1:
    print("Multiple rows with Tick ID '{}' found. Copying TickNet results to all rows...".format(ID))
    print("Rows: ".format(rows))
else:
    print("Tick ID {} found. Copying TickNet results to row...".format(ID))
    pass

net_col = [species, sex, prob]

try:
    NN_Prob = worksheet.find("NN Prob")
    NN_Species = worksheet.find("NN Species")
    NN_Sex = worksheet.find("NN Sex")
except gspread.exceptions.CellNotFound as err:
    print("Column name '{}' was not found in Row 1".format(err))

cols = [NN_Species, NN_Sex, NN_Prob]

row = []
for r in rows:
    # row.append(cell.row)
    for c in range(len(cols)):
        worksheet.update_cell(r.row, cols[c].col, net_col[c])
print(row)


