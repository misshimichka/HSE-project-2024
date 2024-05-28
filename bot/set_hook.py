import requests
import time

TOKEN = "YOUR_TOKEN_HERE"
WEBHOOK_URL = "YOUR_WEBHOOK_HERE"
url = f'https://api.telegram.org/bot{TOKEN}/setWebhook'

params = {'url': WEBHOOK_URL + '123', 'allowed_updates': 'message', 'drop_pending_updates': True}
r = requests.get(url, params=params)
time.sleep(1)
print('set valid')

params = {'url': WEBHOOK_URL, 'allowed_updates': 'message', 'drop_pending_updates': True}
r = requests.get(url, params=params)
print(r.status_code)
print(r.text)
