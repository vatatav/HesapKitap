import os
import json
import logging
import pandas as pd
import fitz  # PyMuPDF için
from datetime import datetime

def pdf_to_jsonl(pdf_path, cutoff_text):
  """PDF dosyasını işler ve metin içeriğini döndürür."""
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
      return pdf_text, pdf_text_size
  except Exception as e:
      logging.error(f"{pdf_path} dosyasını işlerken hata oluştu: {e}")
      return None, 0

def excel_to_jsonl(excel_path):
  """Excel dosyasını işler ve yapılandırılmış veriyi döndürür."""
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

      # datetime nesnelerini string'e dönüştür
      def convert_datetime(obj):
          if isinstance(obj, datetime):
              return obj.strftime('%Y-%m-%d %H:%M:%S')
          return obj

      # DataFrame'leri JSON formatına dönüştür
      info_data = info_df.applymap(convert_datetime).to_dict('records') if info_df is not None else []
      transactions_data = transactions_df.applymap(convert_datetime).to_dict('records') if transactions_df is not None else []

      excel_data = {
          "Bilgiler": info_data,
          "İşlemler": transactions_data
      }

      return excel_data, excel_info_rows, excel_transactions_rows
  except Exception as e:
      logging.error(f"{excel_path} dosyasını işlerken hata oluştu: {e}")
      return None, 0, 0

def validate_jsonl(jsonl_path):
  """JSONL dosyasının formatını kontrol eder."""
  try:
      with open(jsonl_path, 'r', encoding='utf-8') as f:
          for idx, line in enumerate(f, 1):
              entry = json.loads(line)
              if not isinstance(entry, dict):
                  return False, f"Line {idx} is not a dictionary"
              if "messages" not in entry:
                  return False, f"Line {idx} does not contain 'messages' key"
              for msg in entry["messages"]:
                  if not all(key in msg for key in ["role", "content"]):
                      return False, f"Line {idx} has invalid message format"
      return True, "Valid JSONL format"
  except Exception as e:
      return False, f"Error validating JSONL: {str(e)}"

def create_jsonl_for_training(pdf_path, excel_path, cutoff_text, output_jsonl_path):
  """PDF ve Excel dosyalarından eğitim için JSONL dosyası oluşturur."""
  # PDF dosyasını işle
  pdf_text, pdf_text_size = pdf_to_jsonl(pdf_path, cutoff_text)
  if not pdf_text:
      return 0, 0, 0

  # Excel dosyasını işle
  excel_data, excel_info_rows, excel_transactions_rows = excel_to_jsonl(excel_path)
  if not excel_data:
      return 0, 0, 0

  # JSONL dosyasını oluştur
  try:
      with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
          entry = {
              "messages": [
                  {
                      "role": "system",
                      "content": "PDF dosyasından excel formatına dönüştürme yapan bir asistansın."
                  },
                  {
                      "role": "user",
                      "content": pdf_text
                  },
                  {
                      "role": "assistant",
                      "content": json.dumps(excel_data, ensure_ascii=False)
                  }
              ]
          }
          jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

      # JSONL formatını doğrula
      is_valid, validation_message = validate_jsonl(output_jsonl_path)
      if not is_valid:
          raise ValueError(f"Invalid JSONL format: {validation_message}")

      logging.info(f"JSONL dosyası oluşturuldu: {output_jsonl_path}")
      print(f"JSONL dosyası başarıyla oluşturuldu: {output_jsonl_path}")

      return pdf_text_size, excel_info_rows, excel_transactions_rows

  except Exception as e:
      logging.error(f"JSONL dosyası oluşturulurken hata oluştu: {e}")
      return 0, 0, 0