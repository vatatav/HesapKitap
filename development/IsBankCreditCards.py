import os
import logging
from datetime import datetime
from data_processing import create_jsonl_for_training
from fine_tuning import fine_tune_model
from cutoff_utils import get_cutoff_text_from_excel, verify_cutoff_in_pdf
from config_utils import get_api_key, save_model_info

# Development klasörü ayarı
BASE_DIR = "development"

log_file_path = os.path.join(BASE_DIR, "general_operations.log")

logging.basicConfig(
  filename=log_file_path,
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  encoding='utf-8'
)

def create_model_name(model_type):
  """Model adını oluşturur."""
  date_prefix = datetime.now().strftime('%Y-%m-%d')
  model_suffix = model_type.split('-')[1]  # örn: '4o-mini' veya '3.5-turbo'
  return f"{date_prefix}-FirstModel-{model_suffix}"

def process_pdf_files():
  """PDF dosyalarını işler ve JSONL dosyaları oluşturur."""
  pdf_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.pdf')]
  if not pdf_files:
      logging.info("Eğitim dosyası bulunamadı.")
      print("Eğitim dosyası bulunamadı.")
      return []

  successful_samples = []
  total_pdf_text_size = 0
  total_excel_info_rows = 0
  total_excel_transactions_rows = 0

  for pdf_file in pdf_files:
      excel_file = pdf_file.replace('.pdf', '.xlsx')
      pdf_path = os.path.join(BASE_DIR, pdf_file)
      excel_path = os.path.join(BASE_DIR, excel_file)

      # Excel dosyasından cutoff metni oku
      cutoff_text = get_cutoff_text_from_excel(excel_path)
      if cutoff_text == "READ_ERROR":
          logging.error(f"{excel_path} dosyası okunamadı, işleme devam edilemiyor.")
          print(f"{excel_file} okunamadı, işleme devam edilemiyor.")
          continue

      # PDF içinde cutoff metni doğrula
      cutoff_result = verify_cutoff_in_pdf(pdf_path, cutoff_text)
      if cutoff_result in ["READ_ERROR", "TERMINATE"]:
          logging.error(f"{pdf_path} dosyası işlenemedi, işleme devam edilemiyor.")
          print(f"{pdf_file} işlenemedi, işleme devam edilemiyor.")
          continue
      elif cutoff_result == "NO_CUTOFF":
          logging.info(f"{pdf_file} için cutoff metni olmadan devam ediliyor.")
          cutoff_text = None

      # JSONL dosyasını oluştur
      jsonl_path = os.path.join(BASE_DIR, pdf_file.replace('.pdf', '.jsonl'))

      try:
          pdf_text_size, excel_info_rows, excel_transactions_rows = create_jsonl_for_training(
              pdf_path, excel_path, cutoff_text, jsonl_path)
      except Exception as e:
          logging.error(f"{pdf_file} ve {excel_file} işlenirken hata oluştu: {e}")
          print(f"{pdf_file} ve {excel_file} işlenirken hata oluştu, işleme devam edilemiyor.")
          continue

      if pdf_text_size == 0 or excel_transactions_rows == 0:
          logging.error(f"{pdf_file} ve {excel_file} işlenirken hata oluştu, işleme devam edilemiyor.")
          print(f"{pdf_file} ve {excel_file} işlenirken hata oluştu, işleme devam edilemiyor.")
          continue

      successful_samples.append(jsonl_path)
      total_pdf_text_size += pdf_text_size
      total_excel_info_rows += excel_info_rows
      total_excel_transactions_rows += excel_transactions_rows

      logging.info(f"{pdf_file} ve {excel_file} başarıyla işlendi.")

  print(f"{len(successful_samples)} dosya çifti başarıyla işlendi.")
  print(f"Toplam PDF metin boyutu: {total_pdf_text_size} karakter")
  print(f"Toplam 'Bilgiler' satır sayısı: {total_excel_info_rows}")
  print(f"Toplam 'İşlemler' satır sayısı: {total_excel_transactions_rows}")

  return successful_samples

def main():
  # PDF dosyalarını işle ve JSONL dosyaları oluştur
  successful_samples = process_pdf_files()
  
  if not successful_samples:
      logging.error("Hiçbir dosya çifti başarıyla işlenemedi. Eğitim işlemi durduruldu.")
      print("Hiçbir dosya çifti başarıyla işlenemedi. Eğitim işlemi durduruldu.")
      return

  # API anahtarını kontrol et
  api_key = get_api_key()
  if not api_key:
      print("API anahtarı bulunamadı. İşlem sonlandırılıyor.")
      return

  # Model tipini seç
  model_type = input("Model tipi seçin (1. gpt-4o-2024-08-06, 2. gpt-4o-mini-2024-07-18, 3. gpt-3.5-turbo): ")
  selected_model = {
      "1": "gpt-4o-2024-08-06",
      "2": "gpt-4o-mini-2024-07-18",
      "3": "gpt-3.5-turbo"
  }.get(model_type, "gpt-3.5-turbo")

  # Model adı ve açıklaması
  model_name = create_model_name(selected_model)
  model_explanation = input("Model açıklaması: ")

  # Eğitim işlemi
  for jsonl_file in successful_samples:
      model_id = fine_tune_model(
          api_key=api_key,
          jsonl_file=jsonl_file,
          model_type=selected_model,
          model_name=model_name,
          explanation=model_explanation
      )
      
      if model_id:
          logging.info(f"Eğitim işlemi tamamlandı. Model ID: {model_id}")
          print(f"Eğitim işlemi tamamlandı. Model ID: {model_id}")
      else:
          logging.error(f"{jsonl_file} ile model eğitimi başarısız.")

if __name__ == "__main__":
  main()