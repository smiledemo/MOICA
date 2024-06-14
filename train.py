import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from feature_extractor import FeatureExtractor
from domain_classifier import DomainClassifier
from fine_grained_predictor import FineGrainedPredictor
from utils import calculate_loss

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    feature_extractor = FeatureExtractor(model_name=args.model_name).to(device)
    output_dim = feature_extractor.get_output_dim()
    domain_classifiers = [DomainClassifier(output_dim, num_classes=10).to(device) for _ in range(3)]  
    fine_grained_predictor = FineGrainedPredictor(output_dim, num_classes=10).to(device)  

    
    optimizer = optim.Adam(list(feature_extractor.parameters()) +
                           list(fine_grained_predictor.parameters()) +
                           [param for classifier in domain_classifiers for param in classifier.parameters()],
                           lr=args.learning_rate)

    
    criterion = nn.CrossEntropyLoss()

    
    train_loader, val_loader = get_dataloaders(dataset_name=args.dataset, domain=args.domain, batch_size=args.batch_size)

    # 训练
    for epoch in range(args.epochs):
        feature_extractor.train()
        for classifier in domain_classifiers:
            classifier.train()
        fine_grained_predictor.train()

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            
            features = feature_extractor(inputs)
            outputs = [classifier(features) for classifier in domain_classifiers]
            fine_outputs = fine_grained_predictor(features)

            
            loss = sum([calculate_loss(output, labels, criterion) for output in outputs]) / len(outputs)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        feature_extractor.eval()
        for classifier in domain_classifiers:
            classifier.eval()
        fine_grained_predictor.eval()

        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                features = feature_extractor(inputs)
                outputs = [classifier(features) for classifier in domain_classifiers]
                fine_outputs = fine_grained_predictor(features)

                loss = sum([calculate_loss(output, labels, criterion) for output in outputs]) / len(outputs)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}, Val Loss: {val_loss / len(val_loader)}')

    
    torch.save({
        'feature_extractor': feature_extractor.state_dict(),
        'domain_classifiers': [classifier.state_dict() for classifier in domain_classifiers],
        'fine_grained_predictor': fine_grained_predictor.state_dict(),
    }, 'model.pth')
