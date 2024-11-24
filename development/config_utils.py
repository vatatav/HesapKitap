import json
import os
import logging

# Geliştirme ortamında config dosyasının bulunduğu klasör
BASE_DIR = "development"  # TODO: Nihai ortamda bu klasör 'config/' altına taşınacak
config_file_path = os.path.join(BASE_DIR, "config.json")
model_info_path = os.path.join(BASE_DIR, "model_info.json")  # TODO: model bilgileri 'config/model_info.json' altında tutulacak

def load_config():
  """
  Konfigürasyon dosyasını okur ve döndürür.
  """
  try:
      # Önce dosya yolunun doğru olup olmadığını kontrol et
      if not os.path.exists(config_file_path):
          logging.error(f"Konfigürasyon dosyası bulunamadı. Beklenen konum: {os.path.abspath(config_file_path)}")
          return {}

      # Dosya okuma izinlerini kontrol et
      if not os.access(config_file_path, os.R_OK):
          logging.error(f"Konfigürasyon dosyası okuma izni yok: {config_file_path}")
          return {}

      # Dosyayı oku
      with open(config_file_path, 'r', encoding='utf-8') as config_file:
          try:
              config = json.load(config_file)
              logging.info(f"Konfigürasyon dosyası başarıyla yüklendi: {config_file_path}")
              return config
          except json.JSONDecodeError as je:
              logging.error(f"JSON format hatası: {str(je)}")
              logging.error(f"Hata konumu: satır {je.lineno}, sütun {je.colno}")
              logging.error(f"Hatalı karakter: {je.doc[max(0, je.pos-20):je.pos+20]}")
              return {}

  except PermissionError as pe:
      logging.error(f"Dosya erişim izni hatası: {str(pe)}")
      return {}
  except Exception as e:
      logging.error(f"Beklenmeyen hata: {str(e)}")
      logging.error(f"Hata türü: {type(e).__name__}")
      return {}
  
def save_model_info(model_data):
    """
    Model eğitim bilgilerini JSON formatında kaydeder.
    """
    try:
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                if not isinstance(model_info, list):
                    model_info = []
        else:
            model_info = []

        # Yeni model bilgisini en üstte ekle
        model_info.insert(0, model_data)

        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=4, ensure_ascii=False)

        logging.info(f"Model bilgileri kaydedildi: {model_info_path}")
    except Exception as e:
        logging.error(f"Model bilgileri kaydedilemedi: {e}")

def get_api_key():
    """
    OpenAI API anahtarını yükler.
    """
    config = load_config()
    api_key = config.get("api_key", "")
    if not api_key:
        logging.warning("API anahtarı bulunamadı. Lütfen config.json dosyasına ekleyin.")
    return api_key

def get_system_prompt():
  """
  Konfigürasyon dosyasından system prompt'u okur ve döndürür.
  """
  config = load_config()
  system_prompt = config.get("system_prompt", "PDF dosyasından excel formatına dönüştürme yapan bir asistansın.")
  if not system_prompt:
      logging.warning("System prompt bulunamadı. Varsayılan değer kullanılıyor.")
  return system_prompt