"""Module containing neural networks."""
import torch
import torch.nn.functional as F

from .benchmarks import Benchmark
from .layers import (
    AttentionCollapse,
    AverageCollapse,
    CovarianceMatrix,
    Conv,
    NumericalMarkowitz,
    MultiplyByConstant,
    RNN,
    SoftmaxAllocator,
    WeightNorm,
)

from .layers.allocate import NumericalMarkowitzWithShorting


import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.nn.init as init
from .layers.misc import Cov2Corr, CovarianceMatrix, KMeans


class LstmNetFullOpti(torch.nn.Module, Benchmark):
    def __init__(
        self,
        n_assets,
        shrinkage_strategy="diagonal",
        p=0.5,
    ):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(
            1, affine=True
        )
        self.transform_layer = nn.LSTM(
            input_size = n_assets,
            hidden_size = n_assets
            )
        self.covariance_layer = CovarianceMatrix(
            sqrt=False, shrinkage_strategy=shrinkage_strategy
        )
        self.linear = torch.nn.Linear(50 * n_assets, n_assets, bias=True)
        self.portfolio_layer =  UpdNumericalMarkowitzWithShorting(n_assets)

        self.transform_layer.apply(self.init_weights)


    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


    def forward(self, x):
        n_samples = x.shape[0]
        x = self.norm_layer(x)
        # x.shape = (n_samples, 1, lookback, n_assets)

        output, hidden = self.transform_layer(
            x.permute(1, 0, 2, 3)[0] # <-.shape = (n_samples, lookback, n_assets)
        )
        #output.shape = (n_samples, lookback, hidden_size)


        covmat = self.covariance_layer(output)
        exp_rets = torch.tanh(
            self.linear(output.reshape(n_samples, -1))
        )

        weights = self.portfolio_layer(
            exp_rets, covmat
        )

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class RnnNetFullOpti(torch.nn.Module, Benchmark):
    def __init__(
        self,
        n_assets,
        shrinkage_strategy="diagonal",
        p=0.5,
    ):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(
            1, affine=True
        )
        self.transform_layer = nn.RNN(
            input_size = n_assets,
            hidden_size = n_assets,
            dropout = p
            )
        self.covariance_layer = CovarianceMatrix(
            sqrt=False, shrinkage_strategy=shrinkage_strategy
        )
        self.linear = torch.nn.Linear(50 * n_assets, n_assets, bias=True)
        self.portfolio_layer =  UpdNumericalMarkowitzWithShorting(n_assets)


    def forward(self, x):
        n_samples = x.shape[0]
        x = self.norm_layer(x)
        # x.shape = (n_samples, 1, lookback, n_assets)

        output, hidden = self.transform_layer(
            x.permute(1, 0, 2, 3)[0] # <-.shape = (n_samples, lookback, n_assets)
        )
        #output.shape = (n_samples, lookback, hidden_size)


        covmat = self.covariance_layer(output)
        exp_rets = torch.tanh(
            self.linear(output.reshape(n_samples, -1))
        )

        weights = self.portfolio_layer(
            exp_rets, covmat
        )

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class LstmNetMinVar(torch.nn.Module, Benchmark):
    def __init__(
        self,
        n_assets,
        shrinkage_strategy="diagonal",
        p=0.5,
    ):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(
            1, affine=True
        )
        self.transform_layer = nn.LSTM(
            input_size = n_assets,
            hidden_size = n_assets,
            dropout = p
            )
        self.covariance_layer = CovarianceMatrix(
            sqrt=False, shrinkage_strategy=shrinkage_strategy
        )
        self.portfolio_layer = MinVarWithShorting(n_assets)

    def forward(self, x):
        x = self.norm_layer(x)
        # x.shape = (n_samples, 1, lookback, n_assets)

        output, hidden = self.transform_layer(
            x.permute(1, 0, 2, 3)[0] # <-.shape = (n_samples, lookback, n_assets)
        )
        #output.shape = (n_samples, lookback, hidden_size)


        covmat = self.covariance_layer(output)

        weights = self.portfolio_layer(covmat)
        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class RnnNetMinVar(torch.nn.Module, Benchmark):
    def __init__(
        self,
        n_assets,
        shrinkage_strategy="diagonal",
        p=0.5,
    ):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(
            1, affine=True
        )
        self.transform_layer = nn.RNN(
            input_size = n_assets,
            hidden_size = n_assets,
            dropout = p
            )
        self.covariance_layer = CovarianceMatrix(
            sqrt=False, shrinkage_strategy=shrinkage_strategy
        )
        self.portfolio_layer = MinVarWithShorting(n_assets)

    def forward(self, x):
        x = self.norm_layer(x)
        # x.shape = (n_samples, 1, lookback, n_assets)

        output, hidden = self.transform_layer(
            x.permute(1, 0, 2, 3)[0] # <-.shape = (n_samples, lookback, n_assets)
        )
        #output.shape = (n_samples, lookback, hidden_size)


        covmat = self.covariance_layer(output)

        weights = self.portfolio_layer(covmat)
        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }



class ConvNetMinVar(torch.nn.Module, Benchmark):
    def __init__(self, n_channels, lookback, n_assets, p=0.5):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets
        self.cov_n_rows = 50
        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.BatchNorm1d(n_features, affine=True)
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 1), stride=(2, 1), padding=0)

        self.covariance_layer = CovarianceMatrix(
            sqrt=True,
            )
#         self.linear = torch.nn.Linear(120, 5, bias=True)
        self.linear_cov  = torch.nn.Linear(120, 100, bias=True)

        self.portfolio_opt_layer = MinVarWithShorting(n_assets)


    
    

    def forward(self, x):
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError("Input x has incorrect shape {}".format(x.shape))

        n_samples, _, _, _ = x.shape

        # Normalize
        x = x.reshape(n_samples, -1)  # flatten # x.view(n_samples, -1)  # flatten
        x = self.norm_layer(x)
        x = x.reshape(n_samples, 1, self.lookback, self.n_assets)
        conv_res = self.conv_layer(x) # (N, 3, 24, 5)
        pooled = torch.mean(conv_res, dim=1, keepdim=True) # (N, 1, 24, 5)
        
        
        x = self.linear_cov(F.relu(
            pooled.reshape(n_samples, -1)
        ))
        
        covmat = self.covariance_layer(
            x.reshape(n_samples, 20, 5)
        )
        
        
        weights = self.portfolio_opt_layer(
            covmat
        )

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class ConvNetFullOpti(torch.nn.Module, Benchmark):
    def __init__(self, n_channels, lookback, n_assets, p=0.5):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets
        self.cov_n_rows = 50
        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.BatchNorm1d(n_features, affine=True)
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 1), stride=(2, 1), padding=0)

        self.covariance_layer = CovarianceMatrix(
            sqrt=True,
            )
        self.linear = torch.nn.Linear(120, 5, bias=True)
        self.linear_cov  = torch.nn.Linear(120, 100, bias=True)

        self.portfolio_opt_layer = UpdNumericalMarkowitzWithShorting(
            n_assets)


    
    

    def forward(self, x):
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError("Input x has incorrect shape {}".format(x.shape))

        n_samples, _, _, _ = x.shape

        # Normalize
        x = x.reshape(n_samples, -1)  # flatten # x.view(n_samples, -1)  # flatten
        x = self.norm_layer(x)
        x = x.reshape(n_samples, 1, self.lookback, self.n_assets)
       
        conv_res = self.conv_layer(x)
        
        
            
        x_exp_rets = conv_res[:,0,:,:].reshape(n_samples, -1)
        x_cov = conv_res[:,1,:,:].reshape(n_samples, -1)
        
       
        x_cov = F.relu(
            self.linear_cov(x_cov)
        ) 
        x_cov = x_cov.reshape(n_samples, 20, 5)
        
        covmat = self.covariance_layer(x_cov)
        
        exp_rets = torch.tanh(
            self.linear(x_exp_rets)
        )

        weights = self.portfolio_opt_layer(
            exp_rets, covmat
        )

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class MinVarWithShorting(nn.Module):
    def __init__(self, n_assets):
        """Construct."""
        super().__init__()
        covmat_sqrt = cp.Parameter((n_assets, n_assets))


        w = cp.Variable(n_assets)
        risk = cp.sum_squares(covmat_sqrt @ w)

        prob = cp.Problem(
            cp.Maximize(-risk),
            [cp.sum(w) == 1, w >= -1, w <= 1],
        )

        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(
            prob, parameters=[covmat_sqrt], variables=[w]
        )

    def forward(self, covmat_sqrt):
        n_samples, n_assets = covmat_sqrt.shape[0], covmat_sqrt.shape[1]

        return self.cvxpylayer(covmat_sqrt)[0]

class MinVarDenseNet(torch.nn.Module, Benchmark):

    def __init__(self, n_channels, lookback, n_assets, p=0.5):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets
        self.cov_n_rows = 150
        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.BatchNorm1d(n_features, affine=True)
        self.dropout_layer = torch.nn.Dropout(p=p)

        self.linear_for_cov = torch.nn.Linear(n_features, self.n_assets * self.cov_n_rows, bias = True)

        self.covariance_layer = CovarianceMatrix(
            sqrt=True, shrinkage_strategy=None
            )
        self.dropout_layer_cov = torch.nn.Dropout(p=p)


        #self.linear1 = torch.nn.Linear(self.cov_n_rows, n_features, bias=True)

        self.portfolio_opt_layer = MinVarWithShorting (
            n_assets)

    def forward(self, x):
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError("Input x has incorrect shape {}".format(x.shape))

        n_samples, _, _, _ = x.shape

        # Normalize
        x = x.reshape(n_samples, -1)  # flatten # x.view(n_samples, -1)  # flatten
        x = self.norm_layer(x)

      
        y = self.linear_for_cov(x) # (n_samples, n_assets * self.cov_n_rows)
        y = F.relu(y)

      
        y = y.view(n_samples, self.cov_n_rows, -1)  # Reshaping to (n_samples, self.cov_n_rows, n_assets)

      
        y = self.dropout_layer_cov(y)

      
        covmat = self.covariance_layer(y)

        weights = self.portfolio_opt_layer(
            covmat
        )

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class UpdUpdLinearNetMine(torch.nn.Module, Benchmark):
    """Network with one layer.

    Parameters
    ----------
    n_channels : int
        Number of channels, needs to be fixed for each input tensor.

    lookback : int
        Lookback, needs to be fixed for each input tensor.

    n_assets : int
        Number of assets, needs to be fixed for each input tensor.

    p : float
        Dropout probability.

    Attributes
    ----------
    norm_layer : torch.nn.BatchNorm1d
        Batch normalization with learnable parameters.

    dropout_layer : torch.nn.Dropout
        Dropout layer with probability `p`.

    linear : torch.nn.Linear
        One dense layer with `n_assets` outputs and the flattened input tensor `(n_channels, lookback, n_assets)`.

    temperature : torch.Parameter
        Learnable parameter for representing the final softmax allocator temperature.

    allocate_layer : SoftmaxAllocator
        Softmax allocator with a per sample temperature.

    """

    def __init__(self, n_channels, lookback, n_assets, p=0.5):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets
        self.cov_n_rows = 150
        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.BatchNorm1d(n_features, affine=True)
        self.dropout_layer = torch.nn.Dropout(p=p)

        self.linear0 = torch.nn.Linear(n_features, 1000, bias=True)
        self.dropout_layer0 = torch.nn.Dropout(p=p)

        self.linear_for_cov = torch.nn.Linear(n_features, self.n_assets * self.cov_n_rows, bias = True)
        self.covariance_layer = CovarianceMatrix(
            sqrt=True, shrinkage_strategy=None
            )
        self.dropout_layer_cov = torch.nn.Dropout(p=p)


        #self.linear1 = torch.nn.Linear(self.cov_n_rows, n_features, bias=True)
        self.linear1 = torch.nn.Linear(n_features, n_features, bias=True)

        self.dropout_layer1 = torch.nn.Dropout(p=p)
        self.linear2 = torch.nn.Linear(n_features, n_features, bias=True)
        self.dropout_layer2 = torch.nn.Dropout(p=p)
        self.linear = torch.nn.Linear(n_features, n_assets, bias=True)

        self.initialize_weights()


        self.portfolio_opt_layer = UpdNumericalMarkowitzWithShorting (
            n_assets)

    def initialize_weights(self):
      init.kaiming_uniform_(self.linear_for_cov.weight, nonlinearity='relu')
      init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')


    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets). The last 3 dimensions need to be of the same
            size as specified in the constructor. They cannot vary.

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError("Input x has incorrect shape {}".format(x.shape))

        n_samples, _, _, _ = x.shape

        # Normalize
        x = x.reshape(n_samples, -1)  # flatten # x.view(n_samples, -1)  # flatten
        x = self.norm_layer(x)
        #x = self.dropout_layer(x)
        # x = self.linear0(x)
        # x = F.relu(x)
        #x = torch.nn.Dropout(p=0.5)(x)
        #x = self.dropout_layer(x)
        #### Cov computation
        y = self.linear_for_cov(x) # (n_samples, n_assets * self.cov_n_rows)
        y = F.relu(y)
        y = y.view(n_samples, self.cov_n_rows, -1)  # Reshaping to (n_samples, self.cov_n_rows, n_assets)
        y = self.dropout_layer_cov(y)
        covmat = self.covariance_layer(y)
        ####

        # x = self.linear1(x)
        # x = F.relu(x)
        # x = self.dropout_layer1(x)
        # x = self.linear2(x)
        # x = F.relu(x)


        #x = self.dropout_layer2(x)

        x = self.linear(x)
        x = F.torch.tanh(x)
        x = self.dropout_layer(x)
        exp_rets = x#F.relu(x) # (n_samples, n_assets)


        # weights
        weights = self.portfolio_opt_layer(
            exp_rets, covmat
        )

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class UpdDenseNet(torch.nn.Module, Benchmark):
 
    def __init__(self, n_channels, lookback, n_assets, p=0.2):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets
        self.cov_n_rows = 50
        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.BatchNorm1d(n_features, affine=True)
        
   
        self.linear_for_cov = torch.nn.Linear(n_features, self.n_assets * self.cov_n_rows, bias = True)
        self.linear_for_cov_dropout = torch.nn.Dropout(p=p)
        self.covariance_layer = CovarianceMatrix(
            sqrt=True, shrinkage_strategy=None
            )
        

        self.linear = torch.nn.Linear(n_features, n_assets, bias=True)
        self.linear_dropout = torch.nn.Dropout(p=p)

        #self.initialize_weights()

        self.portfolio_opt_layer = NumericalMarkowitzWithShorting (
            n_assets, max_weight=1.5)
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def initialize_weights(self):
      init.kaiming_uniform_(self.linear_for_cov.weight, nonlinearity='relu')
      init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')


    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets). The last 3 dimensions need to be of the same
            size as specified in the constructor. They cannot vary.

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError("Input x has incorrect shape {}".format(x.shape))

        n_samples, _, _, _ = x.shape

    
        x = x.reshape(n_samples, -1)  # flatten # x.view(n_samples, -1)  # flatten
        x = self.norm_layer(x)

      
        y = self.linear_for_cov(x) # (n_samples, n_assets * self.cov_n_rows)
        y = F.relu(y)
        y = y.view(n_samples, self.cov_n_rows, -1)  # Reshaping to (n_samples, self.cov_n_rows, n_assets)
        y = self.linear_for_cov_dropout(y)
        covmat = self.covariance_layer(y)

        x = self.linear(x)
        x = torch.tanh(x)
        x = self.linear_dropout(x)
        exp_rets = x # (n_samples, n_assets)


        # weights
        gamma_sqrt_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype)
            * self.gamma_sqrt
        )
        alpha_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        )

        # weights
        weights = self.portfolio_opt_layer(
            exp_rets, covmat, gamma_sqrt_all, alpha_all
        )

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }

class UpdNumericalMarkowitzWithShorting(nn.Module):
    """Convex optimization layer stylized into portfolio optimization problem.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Attributes
    ----------
    cvxpylayer : CvxpyLayer
        Custom layer used by a third party package called cvxpylayers.

    References
    ----------
    [1] https://github.com/cvxgrp/cvxpylayers

    """

    def __init__(self, n_assets):
        """Construct."""
        super().__init__()
        covmat_sqrt = cp.Parameter((n_assets, n_assets))
        rets = cp.Parameter(n_assets)
        alpha = cp.Parameter(nonneg=True)

        w = cp.Variable(n_assets)
        ret = rets @ w
        risk = cp.sum_squares(covmat_sqrt @ w)

        prob = cp.Problem(
            cp.Maximize(ret - risk),
            [cp.sum(w) == 1, w >= -1, w <= 1],
        )

        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(
            prob, parameters=[rets, covmat_sqrt], variables=[w]
        )

    def forward(self, rets, covmat_sqrt):
        """Perform forward pass.

        Parameters
        ----------
        rets : torch.Tensor
            Of shape (n_samples, n_assets) representing expected returns (or whatever the feature extractor decided
            to encode).

        covmat_sqrt : torch.Tensor
            Of shape (n_samples, n_assets, n_assets) representing the square of the covariance matrix.

        gamma_sqrt : torch.Tensor
            Of shape (n_samples,) representing the tradeoff between risk and return - where on efficient frontier
            we are.

        alpha : torch.Tensor
            Of shape (n_samples,) representing how much L2 regularization is applied to weights. Note that
            we pass the absolute value of this variable into the optimizer since when creating the problem
            we asserted it is going to be nonnegative.

        Returns
        -------
        weights : torch.Tensor
            Of shape (n_samples, n_assets) representing the optimal weights as determined by the convex optimizer.

        """
        n_samples, n_assets = rets.shape

        return self.cvxpylayer(rets, covmat_sqrt)[0]



class UpdLinearNetMine(torch.nn.Module, Benchmark):
    """Network with one layer.

    Parameters
    ----------
    n_channels : int
        Number of channels, needs to be fixed for each input tensor.

    lookback : int
        Lookback, needs to be fixed for each input tensor.

    n_assets : int
        Number of assets, needs to be fixed for each input tensor.

    p : float
        Dropout probability.

    Attributes
    ----------
    norm_layer : torch.nn.BatchNorm1d
        Batch normalization with learnable parameters.

    dropout_layer : torch.nn.Dropout
        Dropout layer with probability `p`.

    linear : torch.nn.Linear
        One dense layer with `n_assets` outputs and the flattened input tensor `(n_channels, lookback, n_assets)`.

    temperature : torch.Parameter
        Learnable parameter for representing the final softmax allocator temperature.

    allocate_layer : SoftmaxAllocator
        Softmax allocator with a per sample temperature.

    """

    def __init__(self, n_channels, lookback, n_assets, p=0.5):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets
        self.cov_n_rows = 150
        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.BatchNorm1d(n_features, affine=True)
        self.dropout_layer = torch.nn.Dropout(p=p)

        self.linear0 = torch.nn.Linear(n_features, 1000, bias=True)
        self.dropout_layer0 = torch.nn.Dropout(p=p)

        self.linear_for_cov = torch.nn.Linear(n_features, self.n_assets * self.cov_n_rows, bias = True)
        self.covariance_layer = CovarianceMatrix(
            sqrt=True, shrinkage_strategy=None
            )
        self.dropout_layer_cov = torch.nn.Dropout(p=p)


        #self.linear1 = torch.nn.Linear(self.cov_n_rows, n_features, bias=True)
        self.linear1 = torch.nn.Linear(n_features, n_features, bias=True)

        self.dropout_layer1 = torch.nn.Dropout(p=p)
        self.linear2 = torch.nn.Linear(n_features, n_features, bias=True)
        self.dropout_layer2 = torch.nn.Dropout(p=p)
        self.linear = torch.nn.Linear(n_features, n_assets, bias=True)

        self.initialize_weights()


        self.portfolio_opt_layer = UpdNumericalMarkowitzWithShorting (
            n_assets)

    def initialize_weights(self):
      init.kaiming_uniform_(self.linear_for_cov.weight, nonlinearity='relu')
      init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')


    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets). The last 3 dimensions need to be of the same
            size as specified in the constructor. They cannot vary.

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError("Input x has incorrect shape {}".format(x.shape))

        n_samples, _, _, _ = x.shape

        # Normalize
        x = x.reshape(n_samples, -1)  # flatten # x.view(n_samples, -1)  # flatten
        x = self.norm_layer(x)
        #x = self.dropout_layer(x)
        # x = self.linear0(x)
        # x = F.relu(x)
        #x = torch.nn.Dropout(p=0.5)(x)
        #x = self.dropout_layer(x)
        #### Cov computation
        y = self.linear_for_cov(x) # (n_samples, n_assets * self.cov_n_rows)
        y = F.relu(y)
        y = y.view(n_samples, self.cov_n_rows, -1)  # Reshaping to (n_samples, self.cov_n_rows, n_assets)
        y = self.dropout_layer_cov(y)
        covmat = self.covariance_layer(y)
        ####

        # x = self.linear1(x)
        # x = F.relu(x)
        # x = self.dropout_layer1(x)
        # x = self.linear2(x)
        # x = F.relu(x)


        #x = self.dropout_layer2(x)

        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        exp_rets = F.relu(x) # (n_samples, n_assets)


        # weights
        weights = self.portfolio_opt_layer(
            exp_rets, covmat
        )

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }



class LinearNetMine(torch.nn.Module, Benchmark):
    """Network with one layer.

    Parameters
    ----------
    n_channels : int
        Number of channels, needs to be fixed for each input tensor.

    lookback : int
        Lookback, needs to be fixed for each input tensor.

    n_assets : int
        Number of assets, needs to be fixed for each input tensor.

    p : float
        Dropout probability.

    Attributes
    ----------
    norm_layer : torch.nn.BatchNorm1d
        Batch normalization with learnable parameters.

    dropout_layer : torch.nn.Dropout
        Dropout layer with probability `p`.

    linear : torch.nn.Linear
        One dense layer with `n_assets` outputs and the flattened input tensor `(n_channels, lookback, n_assets)`.

    temperature : torch.Parameter
        Learnable parameter for representing the final softmax allocator temperature.

    allocate_layer : SoftmaxAllocator
        Softmax allocator with a per sample temperature.

    """

    def __init__(self, n_channels, lookback, n_assets, p=0.5):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets
        self.cov_n_rows = 150
        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.BatchNorm1d(n_features, affine=True)
        self.dropout_layer = torch.nn.Dropout(p=p)

        self.linear0 = torch.nn.Linear(n_features, 1000, bias=True)
        self.dropout_layer0 = torch.nn.Dropout(p=p)

        self.linear_for_cov = torch.nn.Linear(n_features, self.n_assets * self.cov_n_rows, bias = True)
        self.covariance_layer = CovarianceMatrix(
            sqrt=True, shrinkage_strategy=None
            )
        self.dropout_layer_cov = torch.nn.Dropout(p=p)


        #self.linear1 = torch.nn.Linear(self.cov_n_rows, n_features, bias=True)
        self.linear1 = torch.nn.Linear(n_features, n_features, bias=True)

        self.dropout_layer1 = torch.nn.Dropout(p=p)
        self.linear2 = torch.nn.Linear(n_features, n_features, bias=True)
        self.dropout_layer2 = torch.nn.Dropout(p=p)
        self.linear = torch.nn.Linear(n_features, n_assets, bias=True)

        self.temperature = torch.nn.Parameter(
            torch.ones(1), requires_grad=True
        )

        self.portfolio_opt_layer = NumericalMarkowitzWithShorting (
            n_assets)
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)


    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets). The last 3 dimensions need to be of the same
            size as specified in the constructor. They cannot vary.

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError("Input x has incorrect shape {}".format(x.shape))

        n_samples, _, _, _ = x.shape

        # Normalize
        x = x.reshape(n_samples, -1)  # flatten # x.view(n_samples, -1)  # flatten
        x = self.norm_layer(x)
        #x = self.dropout_layer(x)
        # x = self.linear0(x)
        # x = F.relu(x)
        #x = torch.nn.Dropout(p=0.5)(x)
        #x = self.dropout_layer(x)
        #### Cov computation
        y = self.linear_for_cov(x) # (n_samples, n_assets * self.cov_n_rows)
        y = F.relu(y)
        y = y.view(n_samples, self.cov_n_rows, -1)  # Reshaping to (n_samples, self.cov_n_rows, n_assets)
        y = self.dropout_layer_cov(y)
        covmat = self.covariance_layer(y)
        ####

        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout_layer1(x)
        # x = self.linear2(x)
        # x = F.relu(x)


        #x = self.dropout_layer2(x)

        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        exp_rets = F.relu(x) # (n_samples, n_assets)


        # gamma
        gamma_sqrt_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype)
            * self.gamma_sqrt
        )
        alpha_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        )

        # weights
        weights = self.portfolio_opt_layer(
            exp_rets, covmat, gamma_sqrt_all, alpha_all
        )

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class BachelierNetWithShortingUpd(torch.nn.Module, Benchmark):
    """Updated version of BachelierNetWithShorting with 'learnable' cov.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels of the dataset.

    n_assets : int
        Number of assets in our dataset. Note that this network is shuffle invariant along this dimension.

    hidden_size : int
        Hidden state size. Alternatively one can see it as number of output channels.

 
    shrinkage_strategy : str, {'diagonal', 'identity', 'scaled_identity'}
        Strategy of estimating the covariance matrix.

    p : float
        Dropout rate - probability of an element to be zeroed during dropout.

    Attributes
    ----------
    norm_layer : torch.nn.Module
        Instance normalization (per channel).

    transform_layer : deepdow.layers.RNN
        RNN layer that transforms `(n_samples, n_channels, lookback, n_assets)` to
        `(n_samples, hidden_size, lookback, n_assets)` where the first (sample) and the last dimension (assets) is
        shuffle invariant.

    time_collapse_layer : deepdow.layers.AttentionCollapse
        Attention pooling layer that turns  `(n_samples, hidden_size, lookback, n_assets)` into
        `(n_samples, hidden_size, n_assets)` by assigning each timestep in the lookback dimension a weight and
        then performing a weighted average.

    dropout_layer : torch.nn.Module
        Dropout layer where the probability is controled by the parameter `p`.

    covariance_layer : deepdow.layers.CovarianceMatrix
        Estimate square root of a covariance metric for the optimization. Turns `(n_samples, hidden_size, n_assets)` to
        `(n_samples, n_assets, n_assets)`.

    channel_collapse_layer : deepdow.layers.AverageCollapse
        Averaging layer turning `(n_samples, hidden_size, n_assets)` to `(n_samples, n_assets)` where the output
        serves as estimate of expected returns in the optimization.

    gamma : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the tradoff between risk and
        return. If equal to zero only expected returns are considered.

    alpha : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the regularization strength of
        portfolio weights. If zero then no effect if high then encourages weights to be closer to zero.

    portfolio_opt_layer : deepdow.layers.NumericalMarkowitz
        Markowitz optimizer that inputs expected returns, square root of a covariance matrix and a gamma

    """

    def __init__(
        self,
        n_input_channels,
        n_assets,
        hidden_size=32,
        shrinkage_strategy="diagonal",
        p=0.5,
    ):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(
            n_input_channels, affine=True
        )
        self.transform_layer = RNN(n_input_channels, hidden_size=hidden_size)
        self.dropout_layer = torch.nn.Dropout(p=p)
        self.time_collapse_layer = AttentionCollapse(n_channels=hidden_size)
        self.covariance_layer = CovarianceMatrix(
            sqrt=False, shrinkage_strategy=shrinkage_strategy
        )
        self.channel_collapse_layer = AverageCollapse(collapse_dim=1)
        self.portfolio_opt_layer = NumericalMarkowitzWithShorting (
            n_assets)
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    
    
    def get_covmat(self, x):
        """Calculate and return the covariance matrix for a given input.
        
        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets).
            
        Returns
        -------
        covmat : torch.Tensor
            Tensor of shape (n_samples, n_assets, n_assets).
        """
        # Normalize
        x = self.norm_layer(x)

        # Process data through the network up to the point where covariance is calculated
        x = self.transform_layer(x)
        x = self.dropout_layer(x)
        x = self.time_collapse_layer(x)
        covmat = self.covariance_layer(x)
        
        return covmat
    
    
    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        # Normalize
        x = self.norm_layer(x)

        # Covmat
        rets = x[:, 0, :, :]
        # The line below is replaced by taking cov from futher NN output
        # !!! Investigate why did the author used rets of the original data
        # Looks strange
        #covmat = self.covariance_layer(rets) 

        # expected returns
        x = self.transform_layer(x)
        x = self.dropout_layer(x)
        x = self.time_collapse_layer(x)
        covmat = self.covariance_layer(x)

        exp_rets = self.channel_collapse_layer(x)
        

        # gamma
        gamma_sqrt_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype)
            * self.gamma_sqrt
        )
        alpha_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        )

        # weights
        weights = self.portfolio_opt_layer(
            exp_rets, covmat, gamma_sqrt_all, alpha_all
        )

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }



class BachelierNetWithShorting(torch.nn.Module, Benchmark):
    """Combination of recurrent neural networks and convex optimization.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels of the dataset.

    n_assets : int
        Number of assets in our dataset. Note that this network is shuffle invariant along this dimension.

    hidden_size : int
        Hidden state size. Alternatively one can see it as number of output channels.

 
    shrinkage_strategy : str, {'diagonal', 'identity', 'scaled_identity'}
        Strategy of estimating the covariance matrix.

    p : float
        Dropout rate - probability of an element to be zeroed during dropout.

    Attributes
    ----------
    norm_layer : torch.nn.Module
        Instance normalization (per channel).

    transform_layer : deepdow.layers.RNN
        RNN layer that transforms `(n_samples, n_channels, lookback, n_assets)` to
        `(n_samples, hidden_size, lookback, n_assets)` where the first (sample) and the last dimension (assets) is
        shuffle invariant.

    time_collapse_layer : deepdow.layers.AttentionCollapse
        Attention pooling layer that turns  `(n_samples, hidden_size, lookback, n_assets)` into
        `(n_samples, hidden_size, n_assets)` by assigning each timestep in the lookback dimension a weight and
        then performing a weighted average.

    dropout_layer : torch.nn.Module
        Dropout layer where the probability is controled by the parameter `p`.

    covariance_layer : deepdow.layers.CovarianceMatrix
        Estimate square root of a covariance metric for the optimization. Turns `(n_samples, lookback, n_assets)` to
        `(n_samples, n_assets, n_assets)`.

    channel_collapse_layer : deepdow.layers.AverageCollapse
        Averaging layer turning `(n_samples, hidden_size, n_assets)` to `(n_samples, n_assets)` where the output
        serves as estimate of expected returns in the optimization.

    gamma : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the tradoff between risk and
        return. If equal to zero only expected returns are considered.

    alpha : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the regularization strength of
        portfolio weights. If zero then no effect if high then encourages weights to be closer to zero.

    portfolio_opt_layer : deepdow.layers.NumericalMarkowitz
        Markowitz optimizer that inputs expected returns, square root of a covariance matrix and a gamma

    """

    def __init__(
        self,
        n_input_channels,
        n_assets,
        hidden_size=32,
        shrinkage_strategy="diagonal",
        p=0.5,
    ):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(
            n_input_channels, affine=True
        )
        self.transform_layer = RNN(n_input_channels, hidden_size=hidden_size)
        self.dropout_layer = torch.nn.Dropout(p=p)
        self.time_collapse_layer = AttentionCollapse(n_channels=hidden_size)
        self.covariance_layer = CovarianceMatrix(
            sqrt=False, shrinkage_strategy=shrinkage_strategy
        )
        self.channel_collapse_layer = AverageCollapse(collapse_dim=1)
        self.portfolio_opt_layer = NumericalMarkowitzWithShorting (
            n_assets)
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        # Normalize
        x = self.norm_layer(x)

        # Covmat
        rets = x[:, 0, :, :]
        covmat = self.covariance_layer(rets)

        # expected returns
        x = self.transform_layer(x)
        x = self.dropout_layer(x)
        x = self.time_collapse_layer(x)
        exp_rets = self.channel_collapse_layer(x)

        # gamma
        gamma_sqrt_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype)
            * self.gamma_sqrt
        )
        alpha_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        )

        # weights
        weights = self.portfolio_opt_layer(
            exp_rets, covmat, gamma_sqrt_all, alpha_all
        )

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class DummyNet(torch.nn.Module, Benchmark):
    """Minimal trainable network achieving the task.

    Parameters
    ----------
    n_channels : int
        Number of input channels. We learn one constant per channel. Therefore `n_channels=n_trainable_parameters`.
    """

    def __init__(self, n_channels=1):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.mbc = MultiplyByConstant(dim_size=n_channels, dim_ix=1)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        temp = self.mbc(x)
        means = torch.abs(temp).mean(dim=[1, 2]) + 1e-6

        return means / (means.sum(dim=1, keepdim=True))

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class BachelierNet(torch.nn.Module, Benchmark):
    """Combination of recurrent neural networks and convex optimization.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels of the dataset.

    n_assets : int
        Number of assets in our dataset. Note that this network is shuffle invariant along this dimension.

    hidden_size : int
        Hidden state size. Alternatively one can see it as number of output channels.

    max_weight : float
        Maximum weight for a single asset.

    shrinkage_strategy : str, {'diagonal', 'identity', 'scaled_identity'}
        Strategy of estimating the covariance matrix.

    p : float
        Dropout rate - probability of an element to be zeroed during dropout.

    Attributes
    ----------
    norm_layer : torch.nn.Module
        Instance normalization (per channel).

    transform_layer : deepdow.layers.RNN
        RNN layer that transforms `(n_samples, n_channels, lookback, n_assets)` to
        `(n_samples, hidden_size, lookback, n_assets)` where the first (sample) and the last dimension (assets) is
        shuffle invariant.

    time_collapse_layer : deepdow.layers.AttentionCollapse
        Attention pooling layer that turns  `(n_samples, hidden_size, lookback, n_assets)` into
        `(n_samples, hidden_size, n_assets)` by assigning each timestep in the lookback dimension a weight and
        then performing a weighted average.

    dropout_layer : torch.nn.Module
        Dropout layer where the probability is controled by the parameter `p`.

    covariance_layer : deepdow.layers.CovarianceMatrix
        Estimate square root of a covariance metric for the optimization. Turns `(n_samples, lookback, n_assets)` to
        `(n_samples, n_assets, n_assets)`.

    channel_collapse_layer : deepdow.layers.AverageCollapse
        Averaging layer turning `(n_samples, hidden_size, n_assets)` to `(n_samples, n_assets)` where the output
        serves as estimate of expected returns in the optimization.

    gamma : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the tradoff between risk and
        return. If equal to zero only expected returns are considered.

    alpha : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the regularization strength of
        portfolio weights. If zero then no effect if high then encourages weights to be closer to zero.

    portfolio_opt_layer : deepdow.layers.NumericalMarkowitz
        Markowitz optimizer that inputs expected returns, square root of a covariance matrix and a gamma

    """

    def __init__(
        self,
        n_input_channels,
        n_assets,
        hidden_size=32,
        max_weight=1,
        shrinkage_strategy="diagonal",
        p=0.5,
    ):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(
            n_input_channels, affine=True
        )
        self.transform_layer = RNN(n_input_channels, hidden_size=hidden_size)
        self.dropout_layer = torch.nn.Dropout(p=p)
        self.time_collapse_layer = AttentionCollapse(n_channels=hidden_size)
        self.covariance_layer = CovarianceMatrix(
            sqrt=False, shrinkage_strategy=shrinkage_strategy
        )
        self.channel_collapse_layer = AverageCollapse(collapse_dim=1)
        self.portfolio_opt_layer = NumericalMarkowitz(
            n_assets, max_weight=max_weight
        )
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        # Normalize
        x = self.norm_layer(x)

        # Covmat
        rets = x[:, 0, :, :]
        covmat = self.covariance_layer(rets)

        # expected returns
        x = self.transform_layer(x)
        x = self.dropout_layer(x)
        x = self.time_collapse_layer(x)
        exp_rets = self.channel_collapse_layer(x)

        # gamma
        gamma_sqrt_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype)
            * self.gamma_sqrt
        )
        alpha_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        )

        # weights
        weights = self.portfolio_opt_layer(
            exp_rets, covmat, gamma_sqrt_all, alpha_all
        )

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class KeynesNet(torch.nn.Module, Benchmark):
    """Connection of multiple different modules.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels.

    hidden_size : int
        Number of features the transform layer will create.

    transform_type : str, {'RNN', 'Conv'}
        If 'RNN' then one directional LSTM that is shared across all assets. If `Conv` then 1D convolution that
        is shared among all assets.

    n_groups : int
        Number of groups to split the `hidden_size` channels into. This is used in the Group Normalization.
        Note that `hidden_size % n_groups == 0` needs to hold.

    Attributes
    ----------
    norm_layer_1 : torch.nn.InstanceNorm2d
        Instance normalization layer with learnable parameters (2 per channel). Applied to the input.

    transform_layer : torch.nn.Module
        Depends on the `transform_type`. The goal is two exctract features from the input tensor by
        considering the time dimension.

    norm_layer_2 : torch.nn.GroupNorm
        Group normalization with `n_groups` groups. It is applied to the features extracted by `time_collapse_layer`.

    time_collapse_layer, channel_collapse_layer : deepdow.layers.AverageCollapse
        Removing of respective dimensions by the means of averaging.

    temperature : torch.Tensor
        Learnable parameter representing the temperature for the softmax applied to all inputs.

    portfolio_opt_layer : deepdow.layers.SoftmaxAllocator
        Portfolio allocation layer. Uses learned `temperature`.
    """

    def __init__(
        self,
        n_input_channels,
        hidden_size=32,
        transform_type="RNN",
        n_groups=4,
    ):
        self._hparams = locals().copy()
        super().__init__()

        self.transform_type = transform_type

        if self.transform_type == "RNN":
            self.transform_layer = RNN(
                n_input_channels,
                hidden_size=hidden_size,
                bidirectional=False,
                cell_type="LSTM",
            )

        elif self.transform_type == "Conv":
            self.transform_layer = Conv(
                n_input_channels,
                n_output_channels=hidden_size,
                method="1D",
                kernel_size=3,
            )

        else:
            raise ValueError(
                "Unsupported transform_type: {}".format(transform_type)
            )

        if hidden_size % n_groups != 0:
            raise ValueError(
                "The hidden_size needs to be divisible by the n_groups."
            )

        self.norm_layer_1 = torch.nn.InstanceNorm2d(
            n_input_channels, affine=True
        )
        self.temperature = torch.nn.Parameter(
            torch.ones(1), requires_grad=True
        )
        self.norm_layer_2 = torch.nn.GroupNorm(
            n_groups, hidden_size, affine=True
        )
        self.time_collapse_layer = AverageCollapse(collapse_dim=2)
        self.channel_collapse_layer = AverageCollapse(collapse_dim=1)

        self.portfolio_opt_layer = SoftmaxAllocator(temperature=None)

    def __call__(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        n_samples, n_channels, lookback, n_assets = x.shape

        x = self.norm_layer_1(x)
        if self.transform_type == "RNN":
            x = self.transform_layer(x)
        else:
            x = torch.stack(
                [self.transform_layer(x[..., i]) for i in range(n_assets)],
                dim=-1,
            )

        x = self.norm_layer_2(x)
        x = torch.nn.functional.relu(x)
        x = self.time_collapse_layer(x)
        x = self.channel_collapse_layer(x)

        temperatures = (
            torch.ones(n_samples).to(device=x.device, dtype=x.dtype)
            * self.temperature
        )

        weights = self.portfolio_opt_layer(x, temperatures)

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class LinearNet(torch.nn.Module, Benchmark):
    """Network with one layer.

    Parameters
    ----------
    n_channels : int
        Number of channels, needs to be fixed for each input tensor.

    lookback : int
        Lookback, needs to be fixed for each input tensor.

    n_assets : int
        Number of assets, needs to be fixed for each input tensor.

    p : float
        Dropout probability.

    Attributes
    ----------
    norm_layer : torch.nn.BatchNorm1d
        Batch normalization with learnable parameters.

    dropout_layer : torch.nn.Dropout
        Dropout layer with probability `p`.

    linear : torch.nn.Linear
        One dense layer with `n_assets` outputs and the flattened input tensor `(n_channels, lookback, n_assets)`.

    temperature : torch.Parameter
        Learnable parameter for representing the final softmax allocator temperature.

    allocate_layer : SoftmaxAllocator
        Softmax allocator with a per sample temperature.

    """

    def __init__(self, n_channels, lookback, n_assets, p=0.5):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets

        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.BatchNorm1d(n_features, affine=True)
        self.dropout_layer = torch.nn.Dropout(p=p)
        self.linear = torch.nn.Linear(n_features, n_assets, bias=True)

        self.temperature = torch.nn.Parameter(
            torch.ones(1), requires_grad=True
        )
        self.allocate_layer = SoftmaxAllocator(temperature=None)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets). The last 3 dimensions need to be of the same
            size as specified in the constructor. They cannot vary.

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError("Input x has incorrect shape {}".format(x.shape))

        n_samples, _, _, _ = x.shape

        # Normalize
        x = x.view(n_samples, -1)  # flatten
        x = self.norm_layer(x)
        x = self.dropout_layer(x)
        x = self.linear(x)

        temperatures = (
            torch.ones(n_samples).to(device=x.device, dtype=x.dtype)
            * self.temperature
        )
        weights = self.allocate_layer(x, temperatures)

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class MinimalNet(torch.nn.Module, Benchmark):
    """Minimal network that learns per asset weights.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Attributes
    ----------
    allocate_layer : deepdow.allocate.WeightNorm
        Layer whose goal is to learn each weight and make sure they sum up to one via normalization.

    """

    def __init__(self, n_assets):
        super().__init__()
        self.n_assets = n_assets
        self.allocate_layer = WeightNorm(n_assets)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, dim_1, ...., dim_N)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        """
        return self.allocate_layer(x)

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {"n_assets": self.n_assets}


class ThorpNet(torch.nn.Module, Benchmark):
    """All inputs of convex optimization are learnable but do not depend on the input.

    Parameters
    ----------
    n_assets : int
        Number of assets in our dataset. Note that this network is shuffle invariant along this dimension.

    force_symmetric : bool
        If True, then the square root of the covariance matrix will be always by construction symmetric.
        The resulting array will be :math:`M^T M` where :math:`M` is the learnable parameter. If `False` then
        no guarantee of the matrix being symmetric.

    max_weight : float
        Maximum weight for a single asset.


    Attributes
    ----------
    matrix : torch.nn.Parameter
        A learnable matrix of shape `(n_assets, n_assets)`.

    exp_returns : torch.nn.Parameter
        A learnable vector of shape `(n_assets,)`.

    gamma_sqrt : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the tradoff between risk and
        return. If equal to zero only expected returns are considered.

    alpha : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the regularization strength of
        portfolio weights. If zero then no effect if high then encourages weights to be closer to zero.

    """

    def __init__(self, n_assets, max_weight=1, force_symmetric=True):
        self._hparams = locals().copy()
        super().__init__()

        self.force_symmetric = force_symmetric
        self.matrix = torch.nn.Parameter(
            torch.eye(n_assets), requires_grad=True
        )
        self.exp_returns = torch.nn.Parameter(
            torch.zeros(n_assets), requires_grad=True
        )
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        self.portfolio_opt_layer = NumericalMarkowitz(
            n_assets, max_weight=max_weight
        )

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        n = len(x)

        covariance = (
            torch.mm(self.matrix, torch.t(self.matrix))
            if self.force_symmetric
            else self.matrix
        )

        exp_returns_all = torch.repeat_interleave(
            self.exp_returns[None, ...], repeats=n, dim=0
        )
        covariance_all = torch.repeat_interleave(
            covariance[None, ...], repeats=n, dim=0
        )
        gamma_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype)
            * self.gamma_sqrt
        )
        alpha_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        )

        weights = self.portfolio_opt_layer(
            exp_returns_all, covariance_all, gamma_all, alpha_all
        )

        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }
