import pandas as pd
from pathlib import Path

PATH_EXCEL = "data/metadata/punteggi_italy.xlsx"

def main():
    xls = pd.ExcelFile(PATH_EXCEL)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, nrows=2) # Carica solo 2 righe
        print(f"\n📄 FOGLIO: {sheet}")
        print(f"Righe trovate: {len(pd.read_excel(xls, sheet_name=sheet))}")
        # Stampa le colonne con il loro indice per vederle bene
        for i, col in enumerate(df.columns):
            sample = df.iloc[0, i] if not df.empty else "VUOTO"
            print(f"[{i}] {col}  ---> (Esempio: {sample})")

if __name__ == "__main__":
    main()