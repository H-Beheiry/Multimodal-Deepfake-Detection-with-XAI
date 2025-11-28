from captum.attr import Saliency, GradientShap, DeepLift
import torch

def make_forward_func(model):
    def forward_func(x):
        model.eval()
        return model(x)
    return forward_func

def saliency_explination(input_tensor, forward_func):
    saliency= Saliency(forward_func)
    attributions= saliency.attribute(input_tensor, target=1)
    return attributions

def gradshap_explination(input_tensor, forward_func):
    grad_shap= GradientShap(forward_func)
    baseline= torch.zeros_like(input_tensor)
    attributions= grad_shap.attribute(input_tensor, baselines=baseline, target=1)
    return attributions

def deeplift_explanation(input_tensor, forward_func):
    deeplift= DeepLift(forward_func)
    baseline= torch.zeros_like(input_tensor)
    attributions= deeplift.attribute(input_tensor, baselines=baseline, target=1)
    return attributions

class ExplanationPipeline:
    def __init__(self,model):
        self.forward_func= make_forward_func(model)

    def explain(self, input_tensor):
        return {
            "shap": saliency_explination(input_tensor , self.forward_func),
            "gradshap": gradshap_explination(input_tensor , self.forward_func),
            # "deeplift": deeplift_explanation(input_tensor , self.forward_func),
        }

# TODO: add a way to select explainers
# TODO: add More Explination methods