from openai import OpenAI
from config_utils import get_api_key

# API anahtarını yükle ve client oluştur
client = OpenAI(api_key=get_api_key())

# API bağlantısını test et
try:
  response = client.models.list()
  print("API anahtarı başarılı bir şekilde çalışıyor.")
  print("\nKullanılabilir Modeller:")
  print("====================")
  for model in response.data:
      print(f"Model ID: {model.id}")
      print(f"Created: {model.created}")
      print(f"Owner: {model.owned_by}")
      print("-------------------")
except Exception as e:
  print(f"API anahtarı ile ilgili bir hata oluştu: {e}")