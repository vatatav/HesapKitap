import logging
import pandas as pd
import fitz  # PyMuPDF için

def get_cutoff_text_from_excel(excel_path):
    try:
        df_bilgiler = None
        workbook = pd.ExcelFile(excel_path)
        for sheet_name in workbook.sheet_names:
            df = workbook.parse(sheet_name, header=None)
            if 'Cutoff Metni:' in df.values:
                df_bilgiler = df
                break

        if df_bilgiler is not None:
            cutoff_row = df_bilgiler[df_bilgiler[0] == 'Cutoff Metni:']
            if not cutoff_row.empty:
                return cutoff_row[1].values[0]
        logging.warning(f"{excel_path} dosyasında 'Cutoff Metni:' bulunamadı.")
        return None
    except Exception as e:
        logging.error(f"{excel_path} dosyasını okurken hata oluştu: {e}")
        return "READ_ERROR"

def verify_cutoff_in_pdf(pdf_path, cutoff_text):
    try:
        with fitz.open(pdf_path) as doc:
            pdf_text = ''.join(page.get_text() for page in doc)

        if cutoff_text and cutoff_text in pdf_text:
            return cutoff_text
        elif cutoff_text:
            logging.warning(f"{pdf_path} içinde verilen cutoff metni bulunamadı.")
            while True:
                choice = input("1- Yeni cutoff metni gir\n2- Cutoff olmadan devam et\n3- İşlemi kes\nSeçiminiz: ")
                if choice == '1':
                    cutoff_text = input("Yeni cutoff metni girin: ")
                    if cutoff_text in pdf_text:
                        return cutoff_text
                    else:
                        print("Yeni cutoff metni PDF içinde bulunamadı.")
                elif choice == '2':
                    logging.info("Cutoff olmadan devam etme seçildi.")
                    return "NO_CUTOFF"
                elif choice == '3':
                    logging.info("İşlem kesildi.")
                    return "TERMINATE"
                else:
                    print("Lütfen geçerli bir seçim yapınız.")
        else:
            logging.info(f"{pdf_path} dosyasında cutoff metni olmadan devam ediliyor.")
            return "NO_CUTOFF"
    except Exception as e:
        logging.error(f"{pdf_path} dosyasını okurken hata oluştu: {e}")
        return "READ_ERROR"
