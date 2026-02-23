import timm
import torch.nn as nn


def model_preparation(config, num_classes):
    model = timm.create_model('tf_mobilenetv3_small_075', pretrained=False, checkpoint_path=config["model"]["downloaded_model_path"])

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.get_classifier().in_features
    model.classifier = nn.Linear(in_features, num_classes)

    for p in model.get_classifier().parameters():
        p.requires_grad = True

    return model