import torch
import torchxrayvision as xrv

# Load the model
model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")

# Save ONLY the state dictionary (weights)
torch.save(model, "./model/densenet121-res224-mimic_ch_weights.pth")

print("✅ Model weights saved to: densenet121-res224-mimic_ch_weights.pth")
print(f"   State dict type: {type(model)}")  # <class 'dict'>
# import torch
# import torchxrayvision as xrv
# import torchvision.transforms as transforms
# import skimage.io
# import numpy as np

# def load_model(model_path=None, model_type="densenet121-res224-all"):
#     """
#     Load a TorchXRayVision model
    
#     Args:
#         model_path: Path to saved model (can be full model or state_dict)
#         model_type: Model architecture type (used if model_path is None)
    
#     Returns:
#         Loaded model in eval mode
#     """
#     try:
#         if model_path:
#             # Load the checkpoint
#             checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
#             # Check the type of checkpoint
#             if isinstance(checkpoint, dict):
#                 # It's a state_dict
#                 print("📦 Loading from state_dict...")
#                 model = xrv.models.DenseNet(weights=None)
#                 model.load_state_dict(checkpoint)
#             elif hasattr(checkpoint, 'pathologies'):
#                 # It's a full model with the TorchXRayVision interface
#                 print("🤖 Loading full model object...")
#                 model = checkpoint
#             else:
#                 # Try to load as state_dict anyway
#                 print("⚠️ Unknown format, attempting to load as state_dict...")
#                 model = xrv.models.DenseNet(weights=None)
#                 model.load_state_dict(checkpoint)
            
#             print(f"✅ Model loaded from: {model_path}")
#         else:
#             # Load from cache using library's built-in method
#             print("📥 Loading from TorchXRayVision cache...")
#             model = xrv.models.DenseNet(weights=model_type)
#             print(f"✅ Model loaded: {model_type}")
        
#         model.eval()
#         return model
        
#     except Exception as e:
#         print(f"❌ Error loading model: {e}")
#         print("\n💡 Troubleshooting tips:")
#         print("   - If you saved with torch.save(model, path), use torch.load(path, weights_only=False)")
#         print("   - If you saved with torch.save(model.state_dict(), path), use model.load_state_dict()")
#         raise

# def preprocess_xray(image_path, target_size=224):
#     """Preprocess chest X-ray for TorchXRayVision models"""
#     # Read image
#     img = skimage.io.imread(image_path)
    
#     # Handle different image types
#     if len(img.shape) == 3:
#         # RGB image - convert to grayscale
#         img = img.mean(2)
#     elif len(img.shape) == 4:
#         img = img.squeeze()
    
#     # Add channel dimension
#     img = img[None, ...]
    
#     # Normalize to DICOM range [-1024, 1024]
#     # Assuming input is 8-bit (0-255)
#     img = xrv.datasets.normalize(img, 255)
    
#     # Apply TorchXRayVision transforms
#     transform = transforms.Compose([
#         xrv.datasets.XRayCenterCrop(),
#         xrv.datasets.XRayResizer(target_size)
#     ])
    
#     img = transform(img)
#     img = torch.from_numpy(img).float()
    
#     return img

# def predict_diseases(model, image_tensor):
#     """Get disease predictions from preprocessed image"""
#     # Add batch dimension if needed
#     if len(image_tensor.shape) == 3:
#         image_tensor = image_tensor.unsqueeze(0)
    
#     # Ensure correct shape: (batch, channels, height, width)
#     if image_tensor.shape[1] != 1:
#         image_tensor = image_tensor.unsqueeze(1)
    
#     # Get predictions
#     with torch.no_grad():
#         outputs = model(image_tensor)
    
#     # Convert to dictionary
#     predictions = dict(zip(model.pathologies, outputs[0].numpy()))
    
#     return predictions

# def analyze_xray_image(image_path, model_path=None, model_type="densenet121-res224-all"):
#     """Complete pipeline: load model, process image, get predictions"""
#     print("=" * 60)
#     print("CHEST X-RAY ANALYSIS")
#     print("=" * 60)
    
#     # Load model
#     model = load_model(model_path, model_type)
    
#     # Preprocess image
#     print(f"\n📷 Processing image: {image_path}")
#     try:
#         image_tensor = preprocess_xray(image_path)
#         print(f"   Image shape: {image_tensor.shape}")
#     except Exception as e:
#         print(f"❌ Error preprocessing image: {e}")
#         return None
    
#     # Get predictions
#     predictions = predict_diseases(model, image_tensor)
    
#     # Display results
#     print("\n📊 PREDICTIONS:")
#     print("-" * 50)
    
#     sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
#     for disease, score in sorted_preds[:15]:
#         if score >= 0.7:
#             indicator = "🔴 HIGH"
#         elif score >= 0.5:
#             indicator = "🟡 MODERATE"
#         elif score >= 0.3:
#             indicator = "🟢 LOW"
#         else:
#             indicator = "⚪ NEGATIVE"
        
#         print(f"   {disease:35} {score:.4f}  {indicator}")
    
#     return predictions

# def interpret_predictions(predictions):
#     """Convert raw predictions to clinical interpretation"""
#     if predictions is None:
#         return
    
#     print("\n" + "=" * 60)
#     print("CLINICAL INTERPRETATION")
#     print("=" * 60)
    
#     # Target diseases mapping
#     results = {}
    
#     # COPD (Emphysema)
#     emphysema = predictions.get('Emphysema', 0)
#     if emphysema >= 0.7:
#         results['COPD'] = f"High Likelihood ({emphysema:.2%})"
#     elif emphysema >= 0.5:
#         results['COPD'] = f"Moderate Suspicion ({emphysema:.2%})"
#     elif emphysema >= 0.3:
#         results['COPD'] = f"Low Suspicion ({emphysema:.2%})"
#     else:
#         results['COPD'] = f"Unlikely ({emphysema:.2%})"
    
#     # Pneumonia (combine relevant findings)
#     pneumonia_score = max(
#         predictions.get('Pneumonia', 0),
#         predictions.get('Consolidation', 0),
#         predictions.get('Infiltration', 0)
#     )
#     if pneumonia_score >= 0.7:
#         results['Pneumonia'] = f"High Likelihood ({pneumonia_score:.2%})"
#     elif pneumonia_score >= 0.5:
#         results['Pneumonia'] = f"Moderate Suspicion ({pneumonia_score:.2%})"
#     elif pneumonia_score >= 0.3:
#         results['Pneumonia'] = f"Low Suspicion ({pneumonia_score:.2%})"
#     else:
#         results['Pneumonia'] = f"Unlikely ({pneumonia_score:.2%})"
    
#     # Lung Cancer
#     cancer_score = max(predictions.get('Mass', 0), predictions.get('Nodule', 0))
#     if cancer_score >= 0.7:
#         results['Lung Cancer'] = f"High Suspicion - Urgent Follow-up ({cancer_score:.2%})"
#     elif cancer_score >= 0.5:
#         results['Lung Cancer'] = f"Moderate Suspicion - Short-term Follow-up ({cancer_score:.2%})"
#     elif cancer_score >= 0.3:
#         results['Lung Cancer'] = f"Low Suspicion - Routine Follow-up ({cancer_score:.2%})"
#     else:
#         results['Lung Cancer'] = f"Unlikely ({cancer_score:.2%})"
    
#     # Fibrosis
#     fibrosis = predictions.get('Fibrosis', 0)
#     if fibrosis >= 0.7:
#         results['Fibrosis'] = f"High Likelihood ({fibrosis:.2%})"
#     elif fibrosis >= 0.5:
#         results['Fibrosis'] = f"Moderate Suspicion ({fibrosis:.2%})"
#     elif fibrosis >= 0.3:
#         results['Fibrosis'] = f"Low Suspicion ({fibrosis:.2%})"
#     else:
#         results['Fibrosis'] = f"Unlikely ({fibrosis:.2%})"
    
#     print("\n🎯 TARGET DISEASES:")
#     for disease, interpretation in results.items():
#         print(f"   {disease:15}: {interpretation}")
    
#     # Additional notable findings
#     print("\n⚠️ ADDITIONAL NOTABLE FINDINGS:")
#     notable = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pleural_Thickening']
#     for finding in notable:
#         score = predictions.get(finding, 0)
#         if score > 0.5:
#             print(f"   • {finding}: {score:.2%}")

# # ============================================
# # MAIN EXECUTION
# # ============================================

# if __name__ == "__main__":
#     # Path to your image
#     image_path = "./tests/00000001_000.png"
    
#     # Path to your saved model (if you have one)
#     # If you saved the model with torch.save(model, "my_model.pth")
#     model_path = "./model/densenet121_weights.pth"  # Set to your .pth file path if available
#     # model_path = "./my_model.pth"  # Uncomment and set your path
    
#     # Run analysis
#     predictions = analyze_xray_image(image_path, model_path)
    
#     # Get clinical interpretation
#     interpret_predictions(predictions)