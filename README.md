# mahalanobis
mahalanobis by pytorch
A PyTorch implementation of mahalanobis distance 


```python
import torch
import torch.nn as nn


class MahalanobisLayer(nn.Module):
    def __init__(self):
        super(MahalanobisLayer, self).__init__()
        self.mu = None
        self.cov = None
        self.inverse_cov = None

    def forward(self, x):
        """
        Calculates the squared Mahalanobis distance between x and mu
        """
        x = self.to_tensor(x)
        
        self.mu = self.get_mu(x)
        self.cov = self.get_cov(x)
        self.inverse_cov = self.get_inverse_cov(self.cov)
        delta = x - self.mu
        mahalanobis_distance = torch.diag(torch.mm(torch.mm(delta, self.inverse_cov), delta.t()))
        return mahalanobis_distance

    def get_mu(self, x):
        mu = torch.mean(x, dim=0)
        return mu 
    
    def get_cov(self, x):
        """
        Calculates covariance matrix 
        (1/(n - 1)) * Sum((X-mu)^T * (x - mu))

        Reference 
        ---------
        - https://en.wikipedia.org/wiki/Covariance
        """
        n = x.size(0)
        mu = self.get_mu(x)
        delta = x - mu
        cov = (1/(n-1)) * delta.t().mm(delta)
        return cov 

    def get_inverse_cov(self, cov):
        inverse_cov = torch.pinverse(cov)
        return inverse_cov
    
    def to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            if isinstance(data, pd.DataFrame):
                data = data.values
            data = torch.Tensor(data)
        return data
    
    
if __name__ == "__main__":
    from scipy.spatial import distance
    import numpy as np
    import pandas as pd
    
    # sample data 
    normal_samples1 = np.random.normal(loc=0.0, scale=1.0, size=10)
    normal_samples2 = np.random.normal(loc=0.0, scale=1.0, size=10)
    data = pd.DataFrame({"a": normal_samples1, "b": normal_samples2})
    
    #
    # mahalanobis distance using numpy
    #
    inverse_cov = np.linalg.inv(data.cov())
    inverse_cov
    
    mahalanobis_dist = np.diagonal(np.matmul(np.matmul((data - data.mean()).values, inverse_cov), (data - data.mean()).values.T))
    mahalanobis_dist
    
    #
    # mahalanobis distance using MahalanobisLayer
    #
    m_layer = MahalanobisLayer()
    m_layer(data).numpy()
```


Reference:
- https://github.com/bflammers/automahalanobis/blob/master/modules/mahalanobis.py
