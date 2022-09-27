import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CKA import linear_CKA, kernel_CKA
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


class Distance(object):
    """ Module to measure distance between `model1` and `model2`

    Args:
        method: Method to compute distance. 'pwcca' by default.
        model1_names: Names of modules of `model1` to be used. If None (default), all names are used.
        model2_names: Names of modules of `model2` to be used. If None (default), all names are used.
        model1_leaf_modules: Modules of model1 to be considered as single nodes (see https://pytorch.org/blog/FX-feature-extraction-torchvision/).
        model2_leaf_modules: Modules of model2 to be considered as single nodes (see https://pytorch.org/blog/FX-feature-extraction-torchvision/).
        train_mode: If True, models' `train_model` is used, otherwise `eval_mode`. False by default.
    """

    _supported_dims = (2, 4)

    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_names = None,
                 model2_names = None,
                 model1_leaf_modules: list[nn.Module] = None,
                 model2_leaf_modules: list[nn.Module] = None,
                 train_mode: bool = False
                 ):

        dp_ddp = (nn.DataParallel, nn.parallel.DistributedDataParallel)
        if isinstance(model1, dp_ddp) or isinstance(model2, dp_ddp):
            raise RuntimeWarning('model is nn.DataParallel or nn.DistributedDataParallel. '
                                 'SimilarityHook may causes unexpected behavior.')

        self.model1 = model1
        self.model2 = model2
        self.extractor1 = create_feature_extractor(model1, self.convert_names(model1, model1_names,
                                                                              model1_leaf_modules, train_mode))
        self.extractor2 = create_feature_extractor(model2, self.convert_names(model2, model2_names,
                                                                              model2_leaf_modules, train_mode))
        self._model1_tensors: dict[str, torch.Tensor] = None
        self._model2_tensors: dict[str, torch.Tensor] = None

    def available_names(self,
                        model1_leaf_modules: list[nn.Module] = None,
                        model2_leaf_modules: list[nn.Module] = None,
                        train_mode: bool = False
                        ):
        return {'model1': self.convert_names(self.model1, None, model1_leaf_modules, train_mode),
                'model2': self.convert_names(self.model2, None, model2_leaf_modules, train_mode)}

    @staticmethod
    def convert_names(model,
                      names,
                      leaf_modules,
                      train_mode
                      ) -> list[str]:
        # a helper function
        if isinstance(names, str):
            names = [names]
        tracer_kwargs = {}
        if leaf_modules is not None:
            tracer_kwargs['leaf_modules'] = leaf_modules

        _names = get_graph_node_names(model, tracer_kwargs=tracer_kwargs)
        _names = _names[0] if train_mode else _names[1]
        _names = _names[1:]  # because the first element is input

        if names is None:
            names = _names
        else:
            if not (set(names) <= set(_names)):
                diff = set(names) - set(_names)
                raise RuntimeError(f'Unknown names: {list(diff)}')

        return names

    def forward(self,
                data
                ) -> None:
        """ Forward pass of models. Used to store intermediate features.

        Args:
            data: input data to models

        """
        self._model1_tensors = self.extractor1(data)
        self._model2_tensors = self.extractor1(data)

    def between(self,
                name1: str,
                name2: str,
                ) -> torch.Tensor:
        """ Compute distance between modules corresponding to name1 and name2.

        Args:
            name1: Name of a module of `model1`
            name2: Name of a module of `model2`
            size: Size for downsampling if necessary. If size's type is int, both features of name1 and name2 are
            reshaped to (size, size). If size's type is tuple[int, int], features are reshaped to (size[0], size[0]) and
            (size[1], size[1]). If size is None (default), no downsampling is applied.
            downsample_method: Downsampling method: 'avg_pool' for average pooling and 'dft' for discrete
            Fourier transform

        Returns: Distance in tensor.

        """
        tensor1 = self._model1_tensors[name1]
        tensor2 = self._model2_tensors[name2]

        tensor1 = F.interpolate(tensor1, size=256, mode='trilinear', align_corners=False)
        tensor2 = F.interpolate(tensor2, size=256, mode='trilinear', align_corners=False)
        tensor1 = tensor1.squeeze(0)
        tensor2 = tensor2.squeeze(0)
        tensor1 = tensor1.detach().cpu().numpy()
        tensor2 = tensor2.detach().cpu().numpy()
        avg_acts1 = np.mean(tensor1, axis=(1, 2))
        avg_acts2 = np.mean(tensor2, axis=(1, 2))

        cka_dis = linear_CKA(avg_acts1, avg_acts2)
        print(cka_dis)

        return cka_dis