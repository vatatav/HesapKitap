import os
import json
import logging
import pandas as pd
import fitz  # PyMuPDF için
from datetime import datetime

def pdf_to_jsonl(pdf_path, cutoff_text):
    try:
        with fitz.open(pdf_path) as doc:
            pdf_text = ''.join(page.get_text() for page in doc)
        if cutoff_text:
            cutoff_index = pdf_text.find(cutoff_text)
            if cutoff_index != -1:
                pdf_text = pdf_text[:cutoff_index].strip()
            else:
                logging.warning(f"Cutoff metni PDF içinde bulunamadı: {pdf_path}")
        pdf_text_size = len(pdf_text)
        return {"text": pdf_text}, pdf_text_size
    except Exception as e:
        logging.error(f"{pdf_path} dosyasını işlerken hata oluştu: {e}")
        return None, 0

def excel_to_jsonl(excel_path):
    try:
        workbook = pd.ExcelFile(excel_path)
        sheet_names = workbook.sheet_names
        info_df = None
        transactions_df = None

        for sheet in sheet_names:
            df = workbook.parse(sheet)
            columns = df.columns.astype(str).tolist()
            if 'Açıklama' in columns and 'Tutar' in columns:
                transactions_df = df
            elif 'Cutoff Metni:' in df.values:
                info_df = df

        excel_info_rows = info_df.shape[0] if info_df is not None else 0
        excel_transactions_rows = transactions_df.shape[0] if transactions_df is not None else 0

        # datetime nesnelerini string'e dönüştürmek için yardımcı fonksiyon
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(v) for v in obj]
            else:
                return obj

        excel_data = {
            "Bilgiler": convert_datetime(info_df.to_dict(orient='records')) if info_df is not None else [],
            "İşlemler": convert_datetime(transactions_df.to_dict(orient='records')) if transactions_df is not None else []
        }

        return excel_data, excel_info_rows, excel_transactions_rows
    except Exception as e:
        logging.error(f"{excel_path} dosyasını işlerken hata oluştu: {e}")
        return None, 0, 0

def create_jsonl_for_training(pdf_path, excel_path, cutoff_text, output_jsonl_path):
    pdf_jsonl, pdf_text_size = pdf_to_jsonl(pdf_path, cutoff_text)
    if not pdf_jsonl:
        return 0, 0, 0

    excel_jsonl, excel_info_rows, excel_transactions_rows = excel_to_jsonl(excel_path)
    if not excel_jsonl:
        return 0, 0, 0

    # datetime nesnelerini string'e dönüştürmek için yardımcı fonksiyon
    def convert_datetime(obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, dict):
            return {k: convert_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetime(v) for v in obj]
        else:
            return obj

    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        entry = {
            "source": pdf_jsonl["text"],
            "target": excel_jsonl
        }
        # datetime nesnelerini string'e dönüştürüyoruz
        entry = convert_datetime(entry)
        jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logging.info(f"JSONL dosyası oluşturuldu: {output_jsonl_path}")
    print(f"JSONL dosyası başarıyla oluşturuldu: {output_jsonl_path}")

    return pdf_text_size, excel_info_rows, excel_transactions_rows
