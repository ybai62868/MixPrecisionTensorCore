import argparse
import json

torch.backends.cudnn.benchmark = True



NUM_CLASSES = 1000
NUM_WARM_UPS = 50


def get_image_size(model_name):
    image_size_dict = {
        "inception_v3": 299,
        "xception": 299,        
        "nasnetalarge": 331,
        "efficientnet_b5": 456,
    }
    return image_size_dict.get(model_name, 224)

with open("benchmark.json", "r") as f:
    repo_to_models = json.load(f)



def get_model(model_name, pretrained=True, num_classes=NUM_CLASSES):


    return model.cuda().eval()


def main(args):



if __name__ == "__main__":
    main()