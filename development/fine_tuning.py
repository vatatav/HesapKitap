from openai import OpenAI # type: ignore
import logging
import time
import json
from datetime import datetime
from config_utils import save_model_info, get_model_pricing

def fine_tune_model(api_key, jsonl_file, model_type, explanation, epochs=5):
  """OpenAI API kullanarak model eğitimi yapar."""
  client = OpenAI(api_key=api_key)
  try:
      # Dosyayı yükle
      with open(jsonl_file, 'rb') as f:
          file_response = client.files.create(
              file=f,
              purpose='fine-tune'
          )

      # Eğitim başlamadan önce tahmini token ve maliyet hesaplama
      # Bu kısmı IsBankCreditCards.py'den alacağız
      total_tokens = 0
      with open(jsonl_file, 'r', encoding='utf-8') as f:
          for line in f:
              entry = json.loads(line)
              for message in entry["messages"]:
                  content = message["content"]
                  words = content.split()
                  estimated_tokens = 0
                  for word in words:
                      estimated_tokens += len(word) / 3
                      if any(c.isdigit() for c in word):
                          estimated_tokens += 0.5
                      if any(not c.isalnum() for c in word):
                          estimated_tokens += 0.5
                  if message["role"] == "assistant":
                      estimated_tokens *= 1.2
                  total_tokens += estimated_tokens

      estimated_training_tokens = total_tokens * epochs
      pricing = get_model_pricing()
      cost_per_1M = pricing[model_type]["training_cost_per_1M"]
      estimated_cost = (cost_per_1M / 1_000_000) * estimated_training_tokens

      # Fine-tuning işini başlat
      job = client.fine_tuning.jobs.create(
          training_file=file_response.id,
          model=model_type,
          hyperparameters={
              "n_epochs": epochs
          }
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

      # Model bilgilerini kaydet
      training_info = {
          "model_id": job.id,
          "explanation": explanation,
          "model_type": model_type,
          "base_model": model_type,  # İlk eğitimde base_model = model_type
          "epochs": epochs,
          "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          "estimated_training_tokens": estimated_training_tokens,
          "estimated_cost": estimated_cost
      }
      save_model_info(training_info)
      return job.id

  except Exception as e:
      logging.error(f"Model eğitimi sırasında hata oluştu: {e}")
      print(f"Model eğitimi sırasında hata oluştu: {e}")
      return None