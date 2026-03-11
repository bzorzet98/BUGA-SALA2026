import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =============================================================================
# ResNetFishEmbeddings Class: Feature extraction using ResNet and PCA
# =============================================================================

class ResNetFishEmbeddings():
    """
    A class to extract image embeddings using a pre-trained ResNet model 
    and optionally reduce dimensionality using PCA.
    """
    
    def __init__(self, resnet_type="ResNet18", pca_kwargs={}, device="cpu"):
        self.resnet_type = resnet_type
        self.pca_kwargs = pca_kwargs
        self.device = device
        
        # Models
        self.resnet_model = None
        self.standard_scaler = None
        self.pca_model = None
        
        # Initialize
        self._initialize_resnet_model()
        self._initialize_pca()
        
    def _initialize_resnet_model(self, **kwargs):
        # Setup the ResNet architecture and move to the specified device
        if self.resnet_type == "ResNet18":
            full_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Remove the classification head to get embeddings
            self.resnet_model = nn.Sequential(*list(full_model.children())[:-1])
            self.resnet_model.to(self.device)
            self.resnet_model.eval()
        else: 
            print(f"Other ResNet networks initialization not were defined")
            
    def _initialize_pca(self, **kwargs):
        # Initialize Scikit-Learn components for dimensionality reduction
        if isinstance(self.pca_kwargs, dict):
            self.standard_scaler = StandardScaler()
            self.pca_model = PCA(**self.pca_kwargs)
        else:
            print(f"The argument pca_kwargs, is None")
    
    def _fit_PCA_pipeline(self, X_train):
        # Handle conversion from Torch Tensor to Numpy for Scikit-Learn compatibility
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
            if X_train.ndim > 2:
                X_train = X_train.reshape(X_train.shape[0], -1)
        
        # Fit the scaler and PCA on the extracted embeddings
        X_scaled = self.standard_scaler.fit_transform(X_train)
        X_pca = self.pca_model.fit_transform(X_scaled)    
        return X_pca
        
    def fit_model(self, X_train):
        # Extract features using the ResNet backbone and fit PCA parameters
        X_train = X_train.to(self.device)
        
        with torch.no_grad():
            X_embedding = self.resnet_model(X_train)
        
        if self.pca_model is not None:
            return self._fit_PCA_pipeline(X_embedding)
        else:
            return X_embedding
        
    def transform_data(self, X):
        # Apply the learned transformations (ResNet + Scaler + PCA) to new data
        X = X.to(self.device)
        
        with torch.no_grad():
            X_embedding = self.resnet_model(X)
            X_embedding = X_embedding.detach().cpu().numpy()
            if X_embedding.ndim > 2:
                X_embedding = X_embedding.reshape(X_embedding.shape[0], -1)
        
        if self.pca_model is not None:
            X_standard_scale = self.standard_scaler.transform(X_embedding)
            X_pca = self.pca_model.transform(X_standard_scale)
            return X_pca
        else:
            return X_embedding
        
    def save_models_parameters(self, path):
        """
        Serializes and saves the trained scaler and PCA model to a file using torch.save.
        """
        # Create a dictionary containing the state of the Scikit-Learn components
        model_data = {
            'standard_scaler': self.standard_scaler,
            'pca_model': self.pca_model,
            'resnet_type': self.resnet_type,
            'pca_kwargs': self.pca_kwargs
        }
        
        # Save the dictionary to the specified path
        torch.save(model_data, path)

    def load_models_parameters(self, path):
        """
        Loads the scaler and PCA model parameters from a file and updates the instance.
        """
        # Load the data from the disk
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore the Scikit-Learn models and original configuration
        self.standard_scaler = checkpoint.get('standard_scaler')
        self.pca_model = checkpoint.get('pca_model')
        self.resnet_type = checkpoint.get('resnet_type', self.resnet_type)
        self.pca_kwargs = checkpoint.get('pca_kwargs', self.pca_kwargs)
        
        # Re-initialize the ResNet backbone to ensure it matches the loaded type
        self._initialize_resnet_model()