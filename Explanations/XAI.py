from captum.attr import Saliency, GradientShap, InputXGradient, IntegratedGradients, GuidedBackprop, LayerGradCam
import torch.nn.functional as F
import torch

def make_forward_func(model):
    def forward_func(x):
        model.eval()
        return model(x)
    return forward_func

def saliency_explination(input_tensor, forward_func, pred):
    saliency= Saliency(forward_func)
    attributions= saliency.attribute(input_tensor, target=pred)
    return attributions

def integrated_gradients_explination(input_tensor, forward_func, pred):
    ig= IntegratedGradients(forward_func)
    attributions= ig.attribute(
        input_tensor, 
        target=pred, 
        n_steps=10, 
        internal_batch_size=1 
    )
    return attributions

def layer_gradcam_explanation(input_tensor, forward_func, pred, model):
    target_layer = model.backbone[0][-1]
    lgc = LayerGradCam(forward_func, target_layer)
    attributions = lgc.attribute(input_tensor, target=pred, relu_attributions=True)
    target_h, target_w = input_tensor.shape[-2], input_tensor.shape[-1]
    attributions = F.interpolate(
        attributions, 
        size=(target_h, target_w), 
        mode='bilinear', 
        align_corners=False
    )
    return attributions


class ExplanationPipeline:
    def __init__(self,model):
        self.model= model
        self.forward_func= make_forward_func(model)

    def explain(self, input_tensor,pred):
        return {
            "Saliency": saliency_explination(input_tensor , self.forward_func,pred),
            "IntegGrad": integrated_gradients_explination(input_tensor, self.forward_func, pred),
            "Layered GradCAM": layer_gradcam_explanation(input_tensor, self.forward_func, pred, self.model)
        }

# TODO: add a way to select explainers
# TODO: add More Explination methods