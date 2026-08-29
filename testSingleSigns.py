import json
import random
import csv
import os
import re
import unicodedata

fontDir = 'Assurbanipal'  # Neo-Assyrian/Babylonian. Use 'Santakku' for Old Babylonian.
scriptsForFont = {'Assurbanipal': {'NA', 'NB'}, 'Santakku': {'OB'}}[fontDir]

with open('signs_snippets_metadata.json') as f:
    metaData = json.load(f)

# the repo looks prototypes up by the name in prototypes/metadata.csv, so map hex -> that name
hexToName = {}
with open('prototypes/metadata.csv') as f:
    for row in csv.DictReader(f):
        if row['name'] and row[fontDir] == 'True':
            hexToName[row['hex']] = row['name']
skeletons = {fn.split('_')[0] for fn in os.listdir(f'skeletons/{fontDir}') if fn.endswith('_adf.csv')}

subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')


def signHex(signName):
    """eBL sign name (ŠU₂, DIŠ) -> unicode hex (0x122d9) via the Unicode name 'CUNEIFORM SIGN SHU2'."""
    name = signName.strip('|').translate(subscripts).replace('Š', 'SH').replace('š', 'sh')
    name = re.sub(r'\s+', ' ', name.replace('×', ' TIMES ')).strip().upper()
    try:
        symbol = unicodedata.lookup('CUNEIFORM SIGN ' + name)
        return f"0x{ord(symbol):x}", symbol
    except KeyError:
        return None, None


# only sample from entries the model can actually run (right script, known sign, skeleton + image exist)
for entry in metaData:
    entry['hex'], entry['sign'] = signHex(entry['signName'])
    entry['file_path'] = 'singleSignImages/' + entry['_id'] + '.jpeg'
usable = [e for e in metaData
          if e['script'] in scriptsForFont
          and e['hex'] in hexToName and e['hex'] in skeletons
          and os.path.exists(e['file_path'])]
print(f'{len(usable)} of {len(metaData)} entries are usable with {fontDir}')

numberOfSigns = int(input('Enter the number of signs to be tested: '))
indexesToTest = random.sample(range(len(usable)), numberOfSigns)
metaDataToTest = [usable[i] for i in indexesToTest]

outputCsv = f'test_set/singleSigns_{fontDir}.csv'
with open(outputCsv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fn', 'file_path', 'abz', 'name', 'hex', 'sign'])  # columns run_test.py expects
    for entry in metaDataToTest:
        writer.writerow([entry['_id'] + '.jpeg', entry['file_path'], '', hexToName[entry['hex']], entry['hex'], entry['sign']])

print(f'Wrote {numberOfSigns} rows to {outputCsv}. Run all of them with:')
print(f'  python run_test.py --samples_df_path {outputCsv} --font_dir prototypes/{fontDir} '
      f'--con_dir skeletons/{fontDir} --output_folder singleSigns_{fontDir} --ignore_errors')
