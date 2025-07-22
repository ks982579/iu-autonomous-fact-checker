import gradio as gr
from transformers import pipeline
from pathlib import Path

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

    interface = gr.Interface(
        fn=lambda text: classify_text(model, text),
        inputs=gr.Textbox(lines=2, placeholder="Enter Text Here..."),
        outputs="json",
        title="Text Classification with HuggingFace",
        description="This interface uses a HuggingFace model to classify text sentiments. Enter a sentence to see its classification."
    )

    # Launch the Gradio app
    interface.launch()


if __name__ == "__main__":
    main()
