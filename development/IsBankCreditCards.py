import os
import json
import logging
from datetime import datetime
from data_processing import create_jsonl_for_training
from fine_tuning import fine_tune_model
from cutoff_utils import get_cutoff_text_from_excel, verify_cutoff_in_pdf
from config_utils import get_api_key, save_model_info, get_default_epochs, get_model_pricing

# Klasör yapılandırması
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "FineTune")
DEV_DIR = os.path.join(BASE_DIR, "development")
JSONL_FILENAME = "FineTuneIsbank.jsonl"
JSONL_PATH = os.path.join(DATA_DIR, JSONL_FILENAME)
model_info_path = os.path.join(DEV_DIR, "model_info.json")

# Loglama ayarları
log_file_path = os.path.join(DEV_DIR, "general_operations.log")
logging.basicConfig(
  filename=log_file_path,
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  encoding='utf-8'
)

def get_saved_models():
  """Kaydedilmiş model bilgilerini yükler."""
  try:
      if os.path.exists(model_info_path):
          with open(model_info_path, 'r', encoding='utf-8') as f:
              return json.load(f)
  except Exception as e:
      logging.error(f"Model bilgileri yüklenemedi: {e}")
  return []

def process_files():
  """TXT ve Excel dosyalarını işler ve JSONL dosyası oluşturur."""
  # JSONL dosyasının varlığını kontrol et
  JSONL_PATH = os.path.join(DATA_DIR, "FineTuneIsbank.jsonl")

  if os.path.exists(JSONL_PATH):
      print(f"\nMevcut JSONL dosyası bulundu: FineTuneIsbank.jsonl")
      while True:
          choice = input("Bu dosyayı kullanarak model eğitimi yapmak ister misiniz? (E/H): ").upper()
          if choice in ['E', 'H']:
              if choice == 'E':
                  return True, JSONL_PATH
              else:
                  return False, None
          print("Lütfen geçerli bir seçim yapın (E/H)")

  # TXT dosyalarını bul
  txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
  if len(txt_files) < 10:
      logging.error("Yetersiz örnek sayısı. En az 10 örnek gerekli.")
      print("Yetersiz örnek sayısı. En az 10 örnek gerekli. A1")
      print("Txt dosyalrı",txt_files)
      return False, None

  successful_samples = []
  total_txt_size = 0
  total_excel_info_rows = 0
  total_excel_transactions_rows = 0
  first_file = True

  for txt_file in txt_files:
      excel_file = txt_file.replace('.txt', '.xlsx')
      txt_path = os.path.join(DATA_DIR, txt_file)
      excel_path = os.path.join(DATA_DIR, excel_file)

      if not os.path.exists(excel_path):
          logging.warning(f"{excel_file} bulunamadı, bu örnek atlanıyor.")
          continue

      # Excel dosyasından cutoff metni oku
      cutoff_text = get_cutoff_text_from_excel(excel_path)
      if cutoff_text == "READ_ERROR":
          logging.error(f"{excel_path} dosyası okunamadı.")
          continue

      try:
          txt_size, excel_info_rows, excel_transactions_rows = create_jsonl_for_training(
              txt_path, excel_path, cutoff_text, JSONL_PATH, append=not first_file)
          first_file = False
      except Exception as e:
          logging.error(f"{txt_file} ve {excel_file} işlenirken hata oluştu: {e}")
          continue

      if txt_size == 0 or excel_transactions_rows == 0:
          logging.error(f"{txt_file} ve {excel_file} işlenirken hata oluştu.")
          continue

      successful_samples.append((txt_file, excel_file))
      total_txt_size += txt_size
      total_excel_info_rows += excel_info_rows
      total_excel_transactions_rows += excel_transactions_rows

  if len(successful_samples) < 10:
      print(successful_samples)
      logging.error("Yetersiz başarılı örnek sayısı. En az 10 örnek gerekli.")
      print("Yetersiz başarılı örnek sayısı. En az 10 örnek gerekli. A3")
      return False, None

  print(f"\n{len(successful_samples)} dosya çifti başarıyla işlendi:")
  print(f"Toplam metin boyutu: {total_txt_size} karakter")
  print(f"Toplam 'Tablo1' satır sayısı: {total_excel_info_rows}")
  print(f"Toplam 'Tablo2' satır sayısı: {total_excel_transactions_rows}")

  while True:
      choice = input("\nOluşturulan JSONL dosyası ile model eğitimi yapmak ister misiniz? (E/H): ").upper()
      if choice in ['E', 'H']:
          return choice == 'E', JSONL_PATH if choice == 'E' else None
      print("Lütfen geçerli bir seçim yapın (E/H)")

def calculate_estimated_cost(jsonl_path, model_type, epochs):
  """JSONL dosyası için tahmini maliyeti hesaplar."""
  try:
      # Model fiyatlandırmasını al
      pricing = get_model_pricing()
      cost_per_1M = pricing[model_type]["training_cost_per_1M"]

      total_tokens = 0
      with open(jsonl_path, 'r', encoding='utf-8') as f:
          for line in f:
              entry = json.loads(line)
              for message in entry["messages"]:
                  content = message["content"]
                  # Daha gelişmiş token hesaplama
                  # Her kelime ortalama 1.3 token
                  # Her sayı için +1 token
                  # Her özel karakter için +1 token
                  # JSON yapısı için ekstra tokenler
                  words = content.split()
                  estimated_tokens = 0
                  for word in words:
                      # Temel token sayısı
                      estimated_tokens += len(word) / 3  # 3 karakter ≈ 1 token
                      # Sayılar için ek token
                      if any(c.isdigit() for c in word):
                          estimated_tokens += 0.5
                      # Özel karakterler için ek token
                      if any(not c.isalnum() for c in word):
                          estimated_tokens += 0.5

                  # JSON yapısı için ek tokenler
                  if message["role"] == "assistant":
                      estimated_tokens *= 1.2  # JSON yapısı için %20 ek token

                  total_tokens += estimated_tokens

      # Training tokens = input tokens * epoch sayısı
      total_training_tokens = total_tokens * epochs

      # Tahmini maliyeti hesapla
      estimated_cost = (cost_per_1M / 1_000_000) * total_training_tokens

      return total_tokens, estimated_cost
  except Exception as e:
      logging.error(f"Maliyet hesaplama hatası: {e}")
      return 0, 0

def main():
  # Dosyaları işle ve JSONL dosyası oluştur/kontrol et
  proceed_training, jsonl_path = process_files()

  if not proceed_training or not jsonl_path:
      logging.info("Program kullanıcı tercihi ile sonlandırıldı.")
      print("Program sonlandırılıyor.")
      return

  # API anahtarını kontrol et
  api_key = get_api_key()
  if not api_key:
      print("API anahtarı bulunamadı. İşlem sonlandırılıyor.")
      return

  # Model tipini seç
  pricing = get_model_pricing()
  while True:
      print("\nModel tipi seçin:")
      print("1. GPT-4 (En yüksek performans)")
      print("2. GPT-4 Mini (Orta performans)")
      print("3. GPT-3.5 Turbo (Temel performans)")
      print("4. Eğitilmiş model seçimi")
      print("5. Kayıtlı olmayan model")

      choice = input("Seçiminiz (1-5): ")

      if choice == "4":
          saved_models = get_saved_models()
          if not saved_models:
              print("Kayıtlı model bulunamadı.")
              continue

          print("\nKayıtlı modeller:")
          for idx, model in enumerate(saved_models, 1):
              print(f"{idx}. Model ID: {model['model_id']}")
              print(f"   Açıklama: {model['explanation']}")
              print("---")

          print(f"{len(saved_models) + 1}. Geri dön")

          model_choice = input(f"Seçiminiz (1-{len(saved_models) + 1}): ")
          if model_choice.isdigit():
              if int(model_choice) == len(saved_models) + 1:
                  continue
              if 1 <= int(model_choice) <= len(saved_models):
                  selected_model = saved_models[int(model_choice)-1]['model_type']
                  base_model = saved_models[int(model_choice)-1]['model_id']
                  break
      elif choice == "5":
          model_id = input("Model ID'sini girin: ")
          selected_model = input("Model tipini girin: ")
          base_model = model_id
          break
      elif choice.isdigit() and 1 <= int(choice) <= 3:
          selected_model = list(pricing.keys())[int(choice)-1]
          base_model = selected_model
          break

      print("Lütfen geçerli bir seçim yapın.")

  # Epoch sayısını belirle
  default_epochs = get_default_epochs()
  while True:
      epochs = input(f"\nEpoch sayısını girin (varsayılan: {default_epochs}): ")
      epochs = int(epochs) if epochs.isdigit() else default_epochs
      if epochs > 0:
          break
      print("Lütfen geçerli bir sayı girin.")

  # Tahmini maliyeti hesapla ve göster
  total_tokens, estimated_cost = calculate_estimated_cost(jsonl_path, selected_model, epochs)
  print(f"\nTahmini eğitim maliyeti:")
  print(f"Tahmini base token sayısı: {total_tokens:,.2f}")
  print(f"Tahmini toplam training token sayısı: {total_tokens * epochs:,.2f}")
  print(f"Epoch sayısı: {epochs}")
  print(f"Tahmini maliyet: ${estimated_cost:.2f}")
  print("\nNot: Bu maliyet tahminidir. Gerçek maliyet farklılık gösterebilir.")

  # Kullanıcı onayı
  proceed = input("\nBu maliyetle eğitimi başlatmak istiyor musunuz? (E/H): ").upper()
  if proceed != 'E':
      print("İşlem iptal edildi.")
      return

  # Model açıklaması
  model_explanation = input("\nModel açıklaması: ")

  # Eğitim işlemi
  model_id = fine_tune_model(
      api_key=api_key,
      jsonl_file=jsonl_path,
      model_type=selected_model,
      explanation=model_explanation,
      epochs=epochs
  )

  if model_id:
      logging.info(f"Eğitim işlemi tamamlandı. Model ID: {model_id}")
      print(f"\nEğitim işlemi tamamlandı.")
      print(f"Model ID: {model_id}")
  else:
      logging.error("Model eğitimi başarısız.")
      print("\nModel eğitimi başarısız.")

if __name__ == "__main__":
  main()