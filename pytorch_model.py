from PIL import Image

from transformers import ViltProcessor, ViltForQuestionAnswering


def model_pipeline(text: str, image: Image) -> str:
    # prepare image + question
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # text = "How many cats are there?"

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()

    return model.config.id2label[idx]


if __name__ == '__main__':
    model_pipeline(text="Describe what is depicted on the picture?", url="http://images.cocodataset.org/val2017/000000039769.jpg")
