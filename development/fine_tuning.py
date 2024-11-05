from openai import OpenAI
import logging
import time
from datetime import datetime
from config_utils import save_model_info

def calculate_fine_tuning_cost(tokens, model_type):
  """Model türüne göre eğitim maliyetini hesaplar."""
  PRICING = {
      "gpt-4o-2024-08-06": {
          "input": 3.750,
          "cached_input": 1.875,
          "output": 15.000,
          "training": 25.000
      },
      "gpt-4o-mini-2024-07-18": {
          "input": 0.300,
          "cached_input": 0.150,
          "output": 1.200,
          "training": 3.000
      },
      "gpt-3.5-turbo": {
          "input": 3.000,
          "output": 6.000,
          "training": 8.000
      }
  }
  
  costs = {
      "input_cost": (tokens["input"] / 1_000_000) * PRICING[model_type]["input"],
      "output_cost": (tokens["output"] / 1_000_000) * PRICING[model_type]["output"],
      "training_cost": (tokens["training"] / 1_000_000) * PRICING[model_type]["training"]
  }
  if "cached_input" in tokens:
      costs["cached_input_cost"] = (tokens["cached_input"] / 1_000_000) * PRICING[model_type]["cached_input"]
  
  return costs

def fine_tune_model(api_key, jsonl_file, model_type, model_name, explanation):
  """OpenAI API kullanarak model eğitimi yapar."""
  client = OpenAI(api_key=api_key)
  try:
      # Dosyayı yükle
      with open(jsonl_file, 'rb') as f:
          file_response = client.files.create(
              file=f,
              purpose='fine-tune'
          )

      # Fine-tuning işini başlat
      job = client.fine_tuning.jobs.create(
          training_file=file_response.id,
          model=model_type
      )

      logging.info(f"Model eğitimi başlatıldı. Job ID: {job.id}")
      print(f"Model eğitimi başlatıldı. Job ID: {job.id}")

      # Eğitim durumunu takip et
      while True:
          job_status = client.fine_tuning.jobs.retrieve(job.id)
          if job_status.status in ['succeeded', 'failed']:
              break
          time.sleep(60)  # 1 dakika bekle

      if job_status.status == 'failed':
          raise Exception(f"Eğitim başarısız: {job_status.error}")

      # Token kullanımı ve maliyet hesaplama
      tokens = {
          "input": job_status.usage.prompt_tokens,
          "output": job_status.usage.completion_tokens,
          "training": job_status.usage.total_tokens
      }
      costs = calculate_fine_tuning_cost(tokens, model_type)
      total_cost = sum(costs.values())

      # Model bilgilerini kaydet
      training_info = {
          "model_id": job.id,
          "model_name": model_name,
          "explanation": explanation,
          "model_type": model_type,
          "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          "toplam_input_tokens_used": tokens["input"],
          "toplam_output_tokens_used": tokens["output"],
          "toplam_training_tokens_used": tokens["training"],
          "toplam_cost": total_cost,
          "training_history": [{
              "jsonl_file": jsonl_file,
              "file_type": "IsBank",
              "input_tokens_used": tokens["input"],
              "output_tokens_used": tokens["output"],
              "training_tokens_used": tokens["training"],
              "cost": total_cost,
              "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          }]
      }
      save_model_info(training_info)
      return job.id

  except Exception as e:
      logging.error(f"Model eğitimi sırasında hata oluştu: {e}")
      print(f"Model eğitimi sırasında hata oluştu: {e}")
      return None