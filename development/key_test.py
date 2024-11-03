from openai import OpenAI
from config_utils import get_api_key
from datetime import datetime

def format_model_info(model):
  """Model bilgilerini formatlar"""
  created_date = datetime.fromtimestamp(model.created).strftime('%Y-%m-%d')
  return f"Model ID: {model.id}\n" \
         f"Created: {created_date}\n" \
         f"Owner: {model.owned_by}\n" \
         f"-------------------"

# API anahtarını yükle ve client oluştur
client = OpenAI(api_key=get_api_key())

# API çağrısı yap
try:
  response = client.models.list()
  print("API anahtarı başarılı bir şekilde çalışıyor.\n")
  print("Kullanılabilir Modeller:")
  print("====================")
  
  # Modelleri ID'lerine göre sırala
  models = sorted(response.data, key=lambda x: x.id)
  
  # GPT modelleri için filtre
  gpt_models = [model for model in models if 'gpt' in model.id.lower()]
  
  print("\nGPT Modelleri:")
  print("====================")
  for model in gpt_models:
      print(format_model_info(model))
  
  print("\nDiğer Modeller:")
  print("====================")
  other_models = [model for model in models if 'gpt' not in model.id.lower()]
  for model in other_models:
      print(format_model_info(model))

except Exception as e:
  print(f"API anahtarı ile ilgili bir hata oluştu: {e}")