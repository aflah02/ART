import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchviz import make_dot
import os
class MLMForClassification(nn.Module):
    """
    A PyTorch module that performs classification on top of a pre-trained
    Masked Language Model.

    Args:
        hub_model_name (str): The name or path of the pre-trained model.
        ls_activations (list): A list of activation functions for each fully
            connected layer.
        softmax_outputs (bool): Whether to apply a softmax function on the
            output logits.
        fc_dimensions (list): A list of integers that specify the dimensions
            of each fully connected layer.
        out_dimension (int): The dimension of the output tensor.
        ls_layers_to_freeze (list): A list of integers that specify the layers
            to be frozen.
        ls_only_freeze_parameters_having_this_substring (list): A list of
            substrings that specify the names of the parameters to be frozen.
        fn_to_pass_hidden_state_through_before_classification_head (function):
            A function that takes the output value acquired from a forward pass
            through the pre-trained model and returns a tensor that will be
            then passed through the classification head. If None, the last
            hidden state of the CLS token will be used.
        use_backbone_from_cache (bool): Whether to use the backbone from cache if it exists.

    """
    def __init__(self, 
                 hub_model_name, 
                 ls_activations, 
                 softmax_outputs, 
                 fc_dimensions, 
                 out_dimension,
                 ls_layers_to_freeze,
                 ls_only_freeze_parameters_having_this_substring,
                 fn_to_pass_hidden_state_through_before_classification_head,
                 use_backbone_from_cache=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hub_model_name, force_download=not use_backbone_from_cache)
        self.tokenizer = AutoTokenizer.from_pretrained(hub_model_name, force_download=not use_backbone_from_cache)
        self.ls_activations = ls_activations
        self.softmax_outputs = softmax_outputs
        self.fc_dimensions = fc_dimensions
        self.out_dimension = out_dimension
        self.ls_layers_to_freeze = ls_layers_to_freeze
        self.ls_only_freeze_parameters_having_this_substring = ls_only_freeze_parameters_having_this_substring
        self.fn_to_pass_hidden_state_through_before_classification_head = fn_to_pass_hidden_state_through_before_classification_head 
        if self.fn_to_pass_hidden_state_through_before_classification_head is None:
            self.fn_to_pass_hidden_state_through_before_classification_head = self.take_last_hidden_state_cls
            
        self.build_model()

    def build_model(self):
        # freeze layers
        for layer in self.ls_layers_to_freeze:
            for param in self.backbone.base_model.encoder.layer[layer].parameters():
                if self.ls_only_freeze_parameters_having_this_substring is []:
                    param.requires_grad = False
                else:
                    if any([substring in param.name for substring in self.ls_only_freeze_parameters_having_this_substring]):
                        param.requires_grad = False

        # build fc layers
        classification_head = nn.Sequential()
        for i in range(len(self.fc_dimensions)):
            if i == 0:
                classification_head.add_module(f'fc_{i}', nn.Linear(self.backbone.config.hidden_size, self.fc_dimensions[i]))
            else:
                classification_head.add_module(f'fc_{i}', nn.Linear(self.fc_dimensions[i-1], self.fc_dimensions[i]))
            classification_head.add_module(f'activation_{i}', self.ls_activations[i])
        classification_head.add_module(f'fc_{i}_input_{self.fc_dimensions[-1]}_output_{self.out_dimension}', nn.Linear(self.fc_dimensions[-1], self.out_dimension))
        if self.softmax_outputs:
            classification_head.add_module('softmax', nn.Softmax(dim=1))
        self.classification_head = classification_head

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # logits = outputs.logits
        # print(logits.shape)
        # get the last hidden state
        last_hidden_state_cls = self.fn_to_pass_hidden_state_through_before_classification_head(outputs)
        logits = self.classification_head(last_hidden_state_cls)
        return logits
    
    def take_last_hidden_state_cls(self, outputs):
        return outputs[0][:, 0, :]
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def visualize_model(self, save_path):
        os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
        # Plot the architecture
        sentence = 'This is a sentence.'
        input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)])
        attention_mask = torch.tensor([[1] * len(input_ids[0])])
        # make_dot
        model_graph = make_dot(self(input_ids, attention_mask), params=dict(self.named_parameters()))
        model_graph.format = 'png'
        model_graph.render(save_path)


    def print_model(self, save_path):
        with open(save_path, 'w') as f:
            f.write(str(self))
        

if __name__ == "__main__":
    hub_model_name = 'vinai/bertweet-base'
    mlm = MLMForClassification(hub_model_name=hub_model_name,
                                ls_activations=[nn.ReLU(), nn.ReLU()],
                                softmax_outputs=True,
                                fc_dimensions=[100, 50],
                                out_dimension=2,
                                ls_layers_to_freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                ls_only_freeze_parameters_having_this_substring=[],
                                fn_to_pass_hidden_state_through_before_classification_head=None,
                                use_backbone_from_cache=True)
    save_path = hub_model_name.replace('/', '_')
    mlm.visualize_model(f"{save_path}")
    mlm.print_model(f"{save_path}.txt")
    # Comment out the following lines if you don't want to delete the files
    os.remove(f"{save_path}.txt")
    os.remove(f"{save_path}")
    os.remove(f"{save_path}.png")
