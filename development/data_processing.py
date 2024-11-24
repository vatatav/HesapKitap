import os
import json
import logging
import pandas as pd
import fitz  # PyMuPDF için
from datetime import datetime
from config_utils import get_system_prompt  # Import ekleyelim

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
      if info_df is not None:
          # Her sütunu ayrı ayrı dönüştür
          for column in info_df.columns:
              info_df[column] = info_df[column].map(convert_datetime)
          info_data = info_df.to_dict('records')
      else:
          info_data = []

      if transactions_df is not None:
          # Her sütunu ayrı ayrı dönüştür
          for column in transactions_df.columns:
              transactions_df[column] = transactions_df[column].map(convert_datetime)
          transactions_data = transactions_df.to_dict('records')
      else:
          transactions_data = []

      excel_data = {
          "Tablo1": info_data,
          "Tablo2": transactions_data
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

        logging.error(f"JSONL dosyası oluşturulurken hata oluştu: {e}")
        return 0, 0, 0

def create_jsonl_for_training(txt_path, excel_path, cutoff_text, output_jsonl_path, append=False):
  """TXT ve Excel dosyalarından eğitim için JSONL dosyası oluşturur."""
  # System prompt'u config'den al
  system_prompt = get_system_prompt()

  # TXT dosyasını işle
  try:
      with open(txt_path, 'r', encoding='utf-8') as f:
          txt_content = f.read()

      if cutoff_text:
          cutoff_index = txt_content.find(cutoff_text)
          if cutoff_index != -1:
              txt_content = txt_content[:cutoff_index].strip()

      txt_content_size = len(txt_content)
  except Exception as e:
      logging.error(f"{txt_path} dosyasını işlerken hata oluştu: {e}")
      return 0, 0, 0

  # Excel dosyasını işle
  excel_data, excel_info_rows, excel_transactions_rows = excel_to_jsonl(excel_path)
  if not excel_data:
      return 0, 0, 0

  # JSONL dosyasını oluştur/güncelle
  try:
      mode = 'a' if append else 'w'
      with open(output_jsonl_path, mode, encoding='utf-8') as jsonl_file:
          entry = {
              "messages": [
                  {
                      "role": "system",
                      "content": system_prompt  # Config'den gelen system prompt kullanılıyor
                  },
                  {
                      "role": "user",
                      "content": txt_content
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

      logging.info(f"JSONL dosyası güncellendi: {output_jsonl_path}")
      return txt_content_size, excel_info_rows, excel_transactions_rows

  except Exception as e:
      logging.error(f"JSONL dosyası işlenirken hata oluştu: {e}")
      return 0, 0, 0