import gspread

species = 'Ixodes scapularis'
species_prob = '93.63'
sex = 'F'
bloodfed = 'yes'
fed_prob = '80.34'


gc = gspread.service_account(filename='/Users/Lenni/Documents/PycharmProjects/MCEVBD Fellowship/service_account.json')
sh = gc.open("Image_ID")

worksheet = sh.sheet1
column_headers = worksheet.row_values(1)
print(column_headers)

ID = '191005-2139-61'
cells = worksheet.findall(ID, in_column=1)

row = []
net_col = [species, sex, species_prob, bloodfed, fed_prob]
for cell in cells:
    row.append(cell.row)
    for i in range(0, 5):
        worksheet.update_cell(cell.row, i+15, net_col[i])
print(row)


