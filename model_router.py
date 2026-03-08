# Updated model_router.py
# This file implements automatic fallback model handling for response generation

class ModelRouter:
    def __init__(self, models):
        self.models = models

    def generate_response(self, input_data):
        for model in self.models:
            try:
                response = model.generate(input_data)
                return response
            except Exception as e:
                print(f'Error with model {model}: {e}')  # Log the error
        raise RuntimeError('All models failed to generate a response.')  # Raise error if all failed