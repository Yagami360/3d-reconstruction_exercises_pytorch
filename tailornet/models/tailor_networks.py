import torch.nn as nn

# Lists the indices of joints which affect the deformations of particular garment
VALID_THETA = {
    't-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'old-t-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    'pant': [0, 1, 2, 4, 5, 7, 8],
    'skirt' : [0, 1, 2, ],
}

#--------------------------------
# MLP 層
#--------------------------------
class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None):
        super(FullyConnected, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        ]
        for i in range(num_layers - 2):
            net.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
        net.extend([
            nn.Linear(hidden_size, output_size),
        ])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

#--------------------------------
# HF（高周波）メッシュ生成器
#--------------------------------
class HFLayer(nn.Module):
    def __init__(self, params, n_verts = 14154 ):
        super(HFLayer, self).__init__()
        self.params = params
        self.cloth_type = params['garment_class']

        self.mlp = FullyConnected(
            input_size = 72, output_size = n_verts, 
            hidden_size = params['hidden_size'] if 'hidden_size' in params else 1024, 
            num_layers = params['num_layers'] if 'num_layers' in params else 3
        )
        """
        self.mlp = getattr(networks, model_name)(
            input_size=72, output_size=n_verts,
            hidden_size=params['hidden_size'] if 'hidden_size' in params else 1024,
            num_layers=params['num_layers'] if 'num_layers' in params else 3
        )
        """
        return

    def mask_thetas(self, thetas, cloth_type):
        """
        thetas: shape [N, 72]
        cloth_type: e.g. t-shirt
        """
        valid_theta = VALID_THETA[cloth_type]
        mask = torch.zeros_like(thetas).view(-1, 24, 3)
        mask[:, valid_theta, :] = 1.
        mask = mask.view(-1, 72)
        return thetas * mask

    def forward(self, thetas, betas=None, gammas=None):
        thetas = self.mask_thetas(thetas=thetas, cloth_type=self.cloth_type)
        pred_verts = self.mlp(thetas)
        return pred_verts
