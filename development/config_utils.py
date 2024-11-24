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
        with open(config_file_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        logging.info(f"Konfigürasyon dosyası yüklendi: {config_file_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Konfigürasyon dosyası bulunamadı: {config_file_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Konfigürasyon dosyası okunurken hata oluştu: {config_file_path}")
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