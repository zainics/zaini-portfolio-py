import torch
import torch.nn as nn

# 1. Definisi Arsitektur Model PyTorch
class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=1)
        
    def forward(self, x):
        return self.linear(x)

# 2. Inisialisasi Model & Mode Evaluasi
def load_ml_model():
    model = SimpleRegressor()
    model.eval()
    return model

# 3. Fungsi Eksekusi Prediksi / Inferensi
def get_prediction(model: nn.Module, feature_1: float, feature_2: float) -> float:
    input_tensor = torch.tensor([[feature_1, feature_2]], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        return round(float(output.item()), 4)