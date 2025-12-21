from libreyolo import LIBREYOLO

model = LIBREYOLO("libreyolo11n.pt",size="n")
results = model(image="../media/test_image_1_creative_commons.jpg", save=True)