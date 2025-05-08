import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

class BiomassRegressor:
    def __init__(self, conf_thresh=0.66, top_k=50):
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path):
        """Load pretrained FRCNN model"""
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model

    def load_data(self, image_dir, annotation_dir, excel_path):
        """Load and prepare data"""
        # Load Excel ground truths
        excel_df = pd.read_excel(excel_path)
        excel_df["mapped_name"] = excel_df["name"].apply(lambda x: f"Image {int(x.split('_')[-1]) + 1}.jpg")
        self.gt_map = dict(zip(excel_df["mapped_name"], excel_df["Blueberries"]))
        
        # Get unannotated images
        all_images = {f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
        annotated_images = {f.replace(".txt", ".jpg") for f in os.listdir(annotation_dir) if f.endswith(".txt")}
        self.test_images = sorted(list(all_images - annotated_images))

    def predict_counts(self, image_dir):
        """Predict berry counts for all test images"""
        results = []
        
        for filename in tqdm(self.test_images, desc=f"Running model at conf_thresh={self.conf_thresh}"):
            path = os.path.join(image_dir, filename)
            image = Image.open(path).convert("RGB")
            img_tensor = F.to_tensor(image).to(self.device).unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(img_tensor)[0]

            scores = prediction["scores"].cpu()
            boxes = prediction["boxes"].cpu()

            # Filter by confidence threshold and top-k
            keep = scores >= self.conf_thresh
            scores = scores[keep]
            if len(scores) > self.top_k:
                scores = scores[:self.top_k]

            pred_count = len(scores)
            ground_truth = self.gt_map.get(filename, "?")

            results.append({
                "filename": filename,
                "ground_truth_count": ground_truth,
                "predicted_count": pred_count
            })
            
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def simulate_biomass(self, avg_weight=2.8, noise_scale=3.0):
        """Simulate biomass from predicted counts"""
        np.random.seed(42)
        
        def _simulate(count):
            noise = np.random.normal(loc=0.0, scale=noise_scale)
            return avg_weight * count + noise
        
        self.results_df["biomass"] = self.results_df["predicted_count"].apply(_simulate)
        return self.results_df

    def train_regression_models(self):
        """Train and evaluate regression models"""
        # Prepare data
        X = self.results_df[["predicted_count"]].values
        y = self.results_df["biomass"].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet()
        }
        
        # Evaluate models
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                "model": name,
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": mean_squared_error(y_test, y_pred, squared=False),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100
            }
            results.append(metrics)
        
        return pd.DataFrame(results).sort_values(by="R2", ascending=False)

if __name__ == "__main__":
    # Example usage
    regressor = BiomassRegressor(conf_thresh=0.66, top_k=50)
    
    # Load model and data
    regressor.load_model("saved_models/model_finetuned.pth")
    regressor.load_data(
        image_dir="images_blue/final attempt",
        annotation_dir="annotations",
        excel_path="Blueberries.xlsx"
    )
    
    # Get predictions and simulate biomass
    counts_df = regressor.predict_counts("images_blue/final attempt")
    biomass_df = regressor.simulate_biomass()
    
    # Train regression models
    results = regressor.train_regression_models()
    print("\nRegression Model Results:")
    print(results)