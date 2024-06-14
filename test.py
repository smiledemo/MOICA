import torch
from data_loader import get_test_loader
from feature_extractor import FeatureExtractor
from domain_classifier import DomainClassifier
from fine_grained_predictor import FineGrainedPredictor
from utils import spectral_clustering, smooth_labels
import numpy as np

def test_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    feature_extractor = FeatureExtractor(model_name=args.model_name).to(device)
    output_dim = feature_extractor.get_output_dim()
    domain_classifiers = [DomainClassifier(output_dim, num_classes=10).to(device) for _ in range(3)]  
    fine_grained_predictor = FineGrainedPredictor(output_dim, num_classes=10).to(device)  

    
    checkpoint = torch.load('model.pth')
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    for i, classifier in enumerate(domain_classifiers):
        classifier.load_state_dict(checkpoint['domain_classifiers'][i])
    fine_grained_predictor.load_state_dict(checkpoint['fine_grained_predictor'])

    
    test_loader = get_test_loader(dataset_name=args.dataset, domain=args.domain, batch_size=args.batch_size)

    
    feature_extractor.eval()
    for classifier in domain_classifiers:
        classifier.eval()
    fine_grained_predictor.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            
            features = feature_extractor(inputs)
            outputs = [classifier(features) for classifier in domain_classifiers]
            fine_outputs = fine_grained_predictor(features)

           
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    
    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)
    cluster_labels = spectral_clustering(all_features, n_clusters=10)  

    
    smoothed_labels = smooth_labels(cluster_labels)

    
    print(f'Smoothed labels: {smoothed_labels}')
