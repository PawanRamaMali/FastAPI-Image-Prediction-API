from locust import HttpUser, task


class MyUser(HttpUser):
    min_wait = 500
    max_wait = 5000

    @task
    def predict(self):
        with open('img/goldfish.jpg', 'rb') as image:
            self.client.post('/predict', files={'img_file': image})
