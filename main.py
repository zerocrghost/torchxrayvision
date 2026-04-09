# main.py
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
import skimage.io
import numpy as np
import argparse
import sys

def load_model(model_path=None, model_type="densenet121-res224-all"):
    """
    Load a TorchXRayVision model
    
    Args:
        model_path: Path to saved model (can be full model or state_dict)
        model_type: Model architecture type (used if model_path is None)
    
    Returns:
        Loaded model in eval mode
    """
    try:
        if model_path:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Check the type of checkpoint
            if isinstance(checkpoint, dict):
                # It's a state_dict
                model = xrv.models.DenseNet(weights=None)
                model.load_state_dict(checkpoint)
            elif hasattr(checkpoint, 'pathologies'):
                # It's a full model with the TorchXRayVision interface
                model = checkpoint
            else:
                # Try to load as state_dict anyway
                model = xrv.models.DenseNet(weights=None)
                model.load_state_dict(checkpoint)
            
        else:
            # Load from cache using library's built-in method
            model = xrv.models.DenseNet(weights=model_type)
        
        model.eval()
        return model
        
    except Exception as e:
        raise

def preprocess_xray(image_path, target_size=224):
    """Preprocess chest X-ray for TorchXRayVision models"""
    # Read image
    img = skimage.io.imread(image_path)
    
    # Handle different image types
    if len(img.shape) == 3:
        # RGB image - convert to grayscale
        img = img.mean(2)
    elif len(img.shape) == 4:
        img = img.squeeze()
    
    # Add channel dimension
    img = img[None, ...]
    
    # Normalize to DICOM range [-1024, 1024]
    # Assuming input is 8-bit (0-255)
    img = xrv.datasets.normalize(img, 255)
    
    # Apply TorchXRayVision transforms
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(target_size)
    ])
    
    img = transform(img)
    img = torch.from_numpy(img).float()
    
    return img

def predict_diseases(model, image_tensor):
    """Get disease predictions from preprocessed image"""
    # Add batch dimension if needed
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Ensure correct shape: (batch, channels, height, width)
    if image_tensor.shape[1] != 1:
        image_tensor = image_tensor.unsqueeze(1)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Convert to dictionary
    predictions = dict(zip(model.pathologies, outputs[0].numpy()))
    
    return predictions

def analyze_xray_image(image_path, model_path=None, model_type="densenet121-res224-all"):
    """Complete pipeline: load model, process image, get predictions"""
    
    # Load model
    model = load_model(model_path, model_type)
    
    # Preprocess image
    try:
        image_tensor = preprocess_xray(image_path)
    except Exception as e:
        return None
    
    # Get predictions
    predictions = predict_diseases(model, image_tensor)
    
    # Display results
    
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    for disease, score in sorted_preds[:15]:
        if score >= 0.7:
            indicator = "🔴 HIGH"
        elif score >= 0.5:
            indicator = "🟡 MODERATE"
        elif score >= 0.3:
            indicator = "🟢 LOW"
        else:
            indicator = "⚪ NEGATIVE"
        
        print(f"{disease}:{score:.4f}")
    
    return predictions

def interpret_predictions(predictions):
    """Convert raw predictions to clinical interpretation"""
    if predictions is None:
        return
    
    # Target diseases mapping
    results = {}
    
    # COPD (Emphysema)
    emphysema = predictions.get('Emphysema', 0)
    if emphysema >= 0.7:
        results['COPD'] = f"High Likelihood ({emphysema:.2%})"
    elif emphysema >= 0.5:
        results['COPD'] = f"Moderate Suspicion ({emphysema:.2%})"
    elif emphysema >= 0.3:
        results['COPD'] = f"Low Suspicion ({emphysema:.2%})"
    else:
        results['COPD'] = f"Unlikely ({emphysema:.2%})"
    
    # Pneumonia (combine relevant findings)
    pneumonia_score = max(
        predictions.get('Pneumonia', 0),
        predictions.get('Consolidation', 0),
        predictions.get('Infiltration', 0)
    )
    if pneumonia_score >= 0.7:
        results['Pneumonia'] = f"High Likelihood ({pneumonia_score:.2%})"
    elif pneumonia_score >= 0.5:
        results['Pneumonia'] = f"Moderate Suspicion ({pneumonia_score:.2%})"
    elif pneumonia_score >= 0.3:
        results['Pneumonia'] = f"Low Suspicion ({pneumonia_score:.2%})"
    else:
        results['Pneumonia'] = f"Unlikely ({pneumonia_score:.2%})"
    
    # Lung Cancer
    cancer_score = max(predictions.get('Mass', 0), predictions.get('Nodule', 0))
    if cancer_score >= 0.7:
        results['Lung Cancer'] = f"High Suspicion - Urgent Follow-up ({cancer_score:.2%})"
    elif cancer_score >= 0.5:
        results['Lung Cancer'] = f"Moderate Suspicion - Short-term Follow-up ({cancer_score:.2%})"
    elif cancer_score >= 0.3:
        results['Lung Cancer'] = f"Low Suspicion - Routine Follow-up ({cancer_score:.2%})"
    else:
        results['Lung Cancer'] = f"Unlikely ({cancer_score:.2%})"
    
    # Fibrosis
    fibrosis = predictions.get('Fibrosis', 0)
    if fibrosis >= 0.7:
        results['Fibrosis'] = f"High Likelihood ({fibrosis:.2%})"
    elif fibrosis >= 0.5:
        results['Fibrosis'] = f"Moderate Suspicion ({fibrosis:.2%})"
    elif fibrosis >= 0.3:
        results['Fibrosis'] = f"Low Suspicion ({fibrosis:.2%})"
    else:
        results['Fibrosis'] = f"Unlikely ({fibrosis:.2%})"
    
    # for disease, interpretation in results.items():
    #     print(f"{disease}:{interpretation}")
    
    # Additional notable findings
    notable = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pleural_Thickening']
    for finding in notable:
        score = predictions.get(finding, 0)
        # if score > 0.5:
            # print(f"   • {finding}: {score:.2%}")
    
    # Return JSON-friendly results for API
    return {
        "target_diseases": results,
        "notable_findings": {f: predictions.get(f, 0) for f in notable if predictions.get(f, 0) > 0.5},
        "all_predictions": {k: float(v) for k, v in predictions.items()}
    }

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Chest X-ray Analysis using TorchXRayVision')
    parser.add_argument('--image', '-i', 
                       type=str, 
                       required=True,
                       help='Path to the chest X-ray image file')
    parser.add_argument('--model', '-m', 
                       type=str, 
                       default="./model/densenet121-res224-all_weights.pth",
                       help='Path to model weights file (default: ./model/densenet121-res224-all_weights.pth)')
    parser.add_argument('--model-type', '-t', 
                       type=str, 
                       default="densenet121-res224-all",
                       help='Model type if no weights file (default: densenet121-res224-all)')
    parser.add_argument('--json', '-j', 
                       action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Check if image exists
    try:
        import os
        if not os.path.exists(args.image):
            print(f"❌ Error: Image file not found: {args.image}")
            sys.exit(1)
        
        if args.model and not os.path.exists(args.model):
            print(f"⚠️ Warning: Model file not found: {args.model}")
            print("   Will use default model type instead")
            args.model = None
    except Exception as e:
        print(f"❌ Error checking files: {e}")
        sys.exit(1)
    
    # Run analysis
    try:
        predictions = analyze_xray_image(args.image, args.model, args.model_type)
        
        if predictions:
            interpretation = interpret_predictions(predictions)
            sys.exit(0)
        else:
            print("❌ Analysis failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    main()