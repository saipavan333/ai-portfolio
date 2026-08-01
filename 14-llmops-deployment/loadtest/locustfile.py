# Load test the gateway. Run: locust -f loadtest/locustfile.py
from locust import HttpUser, task


class U(HttpUser):
    @task
    def chat(self):
        self.client.post('/v1/chat/completions', json={'model': 'x', 'messages': []})
