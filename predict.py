from cog import BaseModel, Input, Path

from train import train

class Predictor(BaseModel):
            
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

    def predict(
        self,
        input_images: Path = Input(
            description="A .zip or .tar file containing the image files that will be used for fine-tuning"
        ),
        use_face_detection_instead: bool = Input(
            description="If you want to use face detection instead of CLIPSeg for masking. For face applications, we recommend using this option.",
            default=True,
        ),
    ) -> Path:
        
        training_result = train(input_images,use_face_detection_instead)
    

        
        return training_result.weights