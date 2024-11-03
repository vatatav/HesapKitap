# fine_tuning.py
from openai import OpenAI
import logging
import json
import os
from datetime import datetime
from config_utils import save_model_info

def calculate_fine_tuning_cost(training_tokens, model_type):
  """
  Model türüne göre eğitim maliyetini hesaplar.
  """
  PRICING = {
      "gpt-4o-2024-08-06": {"training": 25.000},
      "gpt-4o-mini-2024-07-18": {"training": 3.000},
      "gpt-3.5-turbo": {"training": 8.000},
  }
  price_per_million = PRICING[model_type]["training"]
  return (training_tokens / 1_000_000) * price_per_million

def save_training_results(model_id, training_info):
  """
  Eğitim sonuçlarını model_info.json dosyasına kaydeder.
  """
  save_model_info(training_info)

def fine_tune_model(api_key, jsonl_file, model_type, simulate=True):
  """
  OpenAI API kullanarak model eğitimi yapar veya simüle eder.
  """
  if simulate:
      logging.info(f"Simülasyon modunda model eğitimi yapılıyor. Model tipi: {model_type}")
      print(f"Simülasyon: {jsonl_file} dosyası ile {model_type} modelinin eğitimi yapılıyor.")
      training_tokens = 10000
      cost = calculate_fine_tuning_cost(training_tokens, model_type)
      model_id = "simulated_model_id"

      training_info = {
          "model_id": model_id,
          "jsonl_file": jsonl_file,
          "model_type": model_type,
          "training_tokens_used": training_tokens,
          "total_cost": cost,
          "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      }
      save_training_results(model_id, training_info)
      return model_id

  client = OpenAI(api_key=api_key)
  try:
      # Dosyayı yükle
      with open(jsonl_file, 'rb') as f:
          file_response = client.files.create(
              file=f,
              purpose='fine-tune'
          )

      # Fine-tuning işlemini başlat
      fine_tune_response = client.fine_tuning.jobs.create(
          training_file=file_response.id,
          model=model_type
      )

      model_id = fine_tune_response.id
      logging.info(f"Gerçek model eğitimi başlatıldı. Model ID: {model_id}")
      print(f"Gerçek model eğitimi başlatıldı. Model ID: {model_id}")

      # Token kullanımını ve maliyeti hesapla
      training_tokens = 10000  # Gerçek token sayısını API'den alabilirsiniz
      cost = calculate_fine_tuning_cost(training_tokens, model_type)

      training_info = {
          "model_id": model_id,
          "jsonl_file": jsonl_file,
          "model_type": model_type,
          "training_tokens_used": training_tokens,
          "total_cost": cost,
          "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      }
      save_training_results(model_id, training_info)

      return model_id
  except Exception as e:
      logging.error(f"Model eğitimi sırasında hata oluştu: {e}")
      print(f"Model eğitimi sırasında hata oluştu: {e}")
      return None