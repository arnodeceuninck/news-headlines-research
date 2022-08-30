# The functions used here are based on code written in "0 Introduction.ipynb",
#  look at the notebook for extra explanation.

import pandas as pd
import openpyxl  # Dependency of pandas that's not automatically installed


def get_useless_columns():
    return ["Wat zit erin voor mij?", "Modaliteit", "Sensatie"]


def get_preprocessed_dataset():
    # Dataset
    df = pd.read_excel('headline-data/Dataverwerking.xlsx', sheet_name='Verwerking')

    # Column names
    df.columns = [x.strip() for x in df.columns]

    # Test ID's
    for i in range(1, len(df)):
        if pd.isna(df.loc[i, 'Test']) or str(df.loc[i, 'Test']).strip() == '':  # Strip required for e.g. test 49.D
            df.loc[i, 'Test'] = df.loc[i - 1, 'Test']

    df = df.drop(columns=get_useless_columns())

    return df


import urllib, json
from IPython.display import Markdown, display

# Note: Couldn't get the notebooks working to give their file path to this function (__file__ didn't work)
def generate_toc(notebook_path, indent_char="&emsp;"):
    # Generate a table of contents for a notebook.
    # Code entirely from StackOverflow: https://stackoverflow.com/questions/21151450/how-can-i-add-a-table-of-contents-to-a-jupyter-jupyterlab-notebook
    # Note: Not entirely from that site, I made a small change to make it immediately display the generated table of contents.

    # Get the file of the current notebook
    is_markdown = lambda it: "markdown" == it["cell_type"]
    is_title = lambda it: it.strip().startswith("#") and it.strip().lstrip("#").lstrip()
    with open(notebook_path, 'r') as in_f:
        nb_json = json.load(in_f)
    for cell in filter(is_markdown, nb_json["cells"]):
        for line in filter(is_title, cell["source"]):
            line = line.strip()
            indent = indent_char * (line.index(" ") - 1)
            title = line.lstrip("#").lstrip()
            url = urllib.parse.quote(title.replace(" ", "-"))
            out_line = f"{indent}[{title}](#{url})<br>\n"

            # Choose one of the two lines below, the other should be commented
            print(out_line, end="") # This just prints it to the console
            display(Markdown(out_line)) # Line I added for Ipython markdown display
