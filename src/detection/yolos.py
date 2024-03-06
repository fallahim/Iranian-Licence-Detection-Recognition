import json
from PIL import Image
from huggingface_hub import hf_hub_download
import yolov5

class LicensePlateDetector:
    """
    A class for detecting license plates in images using YOLOv5 model.
    """

    def __init__(self, image_path, model_ids):
        """
        Initialize the LicensePlateDetector object.

        Parameters:
        - image_path (str): The path to the input image.
        - model_ids (list): List of model identifiers to choose from.
        """
        self.image_path = image_path
        self.app_title = "License Plate Object Detection"
        self.model_ids = model_ids
        self.current_model_id = model_ids[-1]
        self.model = yolov5.load(self.current_model_id)
        #TODO: aadd exapmles

    def predict(self, threshold=0.6, model_id=None):
        """
        Perform license plate detection on the input image.

        Parameters:
        - threshold (float): Confidence threshold for object detection.
        - model_id (str): Identifier for the YOLOv5 model to be used.

        Returns:
        - PIL.Image.Image: Image object with detected license plates.
        """
        if model_id != self.current_model_id:
            self.model = yolov5.load(model_id)
            self.current_model_id = model_id

        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        input_size = config["input_size"]

        self.model.conf = threshold
        image = Image.open(self.image_path)
        results = self.model(image, size=input_size)
        numpy_image = results.render()[0]
        output_image = Image.fromarray(numpy_image)
        return output_image

    def save_predicted_image(self, output_path, threshold=0.6, model_id=None):
        """
        Save the predicted image after license plate detection.

        Parameters:
        - output_path (str): The path where the predicted image will be saved.
        - threshold (float): Confidence threshold for object detection.
        - model_id (str): Identifier for the YOLOv5 model to be used.
        """
        predicted_image = self.predict(threshold, model_id)
        predicted_image.save(output_path)

if __name__ == "__main__":
    image_path = r'./data/images/13.jpg'
    models_ids = ['keremberke/yolov5n-license-plate', 'keremberke/yolov5s-license-plate', 'keremberke/yolov5m-license-plate']
    detector = LicensePlateDetector(image_path, models_ids)
    output_path = "test.jpg"
    detector.save_predicted_image(output_path, threshold=0.6, model_id=models_ids[-1])
