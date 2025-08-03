# import gradio as gr
from transformers import pipeline
from pathlib import Path

class ClaimExtractor():
    __class_name__ = "ClaimExtractor"

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        model_path = current_dir / 'trainingresults' / 'latest'

        self._model = pipeline(
            task="text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device='cuda'
        )
    
    def assign_claim_score_to_text(self, sentences: str):
        """
        The string parameter can be multiple sentences, like a post from social media.
        This function will separate based on sentences and determine if they are claims.
        It will return a list of results

        Args:
            text (str): _description_
        """
        # TODO: Better typing
        claims = []
        # TODO: Could probably imporve sentence parsing
        sentence_list = sentences.split('. ')
        for sentence in sentence_list:
            # TODO: Possible adjustments
            if len(sentence.strip()) > 10: # filter out short statements
                result = self.classify_text(sentence)
                claims.append({
                    'text': sentence,
                    # Cound be enum?
                    ## OR -> Fact% = 1- Fake% => Might not be the same thing
                    'label': 'Claim' if result[0]['label'] == 'LABEL_1' else 'Opinion',
                    'confidence': result[0]['score']
                })
        return claims
    
    ## TODO: validate input
    ## TODO: typing
    def classify_text(self, text: str):
        return self._model(text)
        
# out of class
def load_model():
    model_path = Path().cwd() / 'trainingresults' /'latest'
    assert model_path.exists()
    classifier = pipeline(
        task="text-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        device='cuda'
    )
    return classifier

def classify_text(model, text):
    result = model(text)
    print(result)
    return result

def main():
    model = load_model()

    # interface = gr.Interface(
    #     fn=lambda text: classify_text(model, text),
    #     inputs=gr.Textbox(lines=2, placeholder="Enter Text Here..."),
    #     outputs="json",
    #     title="Text Classification with HuggingFace",
    #     description="This interface uses a HuggingFace model to classify text sentiments. Enter a sentence to see its classification."
    # )

    # # Launch the Gradio app
    # interface.launch()


if __name__ == "__main__":
    main()
