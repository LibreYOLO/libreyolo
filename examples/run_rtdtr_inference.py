from libreyolo import LIBREYOLO

model = LIBREYOLO("librertdetrs.pth", size="s")

result = model("/Users/xuban.ceccon/Documents/GitHub/libreyolo/media/parkour.jpg", save=True, output_path="/Users/xuban.ceccon/Documents/GitHub/libreyolo/examples/testazo")


