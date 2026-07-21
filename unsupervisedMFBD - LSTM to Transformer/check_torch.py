import torch

print(f"Versión de PyTorch: {torch.__version__}")

# Crear un 'tensor' (la unidad básica de PyTorch, como una matriz)
tensor = torch.rand(3, 3)

print("\nTu primer Tensor (matriz de números aleatorios):")
print(tensor)

# Comprobar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nEjecutando en: {device}")