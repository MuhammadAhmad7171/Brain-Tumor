import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, test_dataset, device):
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Metrics
    test_accuracy = np.mean(np.array(test_preds) == np.array(test_labels)) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(classification_report(test_labels, test_preds, target_names=test_dataset.classes))

    # Confusion Matrix
    conf_matrix = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=test_dataset.classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix on Test Dataset')
    plt.show()
