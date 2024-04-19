from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-infographics-vqa-large").to("cuda")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-infographics-vqa-large")

def get_answer(image, question):
    # image = Image.open("/home/ges/level3-cv-productserving-cv-10/data/infographics/images/36919.jpeg")
    # question = "What percentage of Canadian internet users search for online health information not from home"
    inputs = processor(images=image, text=question, return_tensors="pt").to("cuda")
    # ins = processor(images = image,text=question,return_tensors='pt').to('cuda')
    predictions = model.generate(**inputs)
    pred = processor.decode(predictions[0], skip_special_tokens=True)
    return pred
