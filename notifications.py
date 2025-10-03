# notifications.py
import requests
import config

class TelegramNotifier:
    def __init__(self):
        self.token = getattr(config, 'TELEGRAM_TOKEN', None)
        self.chat_id = getattr(config, 'TELEGRAM_CHAT_ID', None)
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, message):
        if not self.token or not self.chat_id:
            return

        payload = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'Markdown'}
        try:
            requests.post(self.base_url, json=payload)
        except Exception as e:
            print(f"Error sending Telegram message: {e}")