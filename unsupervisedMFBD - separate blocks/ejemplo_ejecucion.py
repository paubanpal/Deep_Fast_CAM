# Instanciamos la red completa predimensionada para predecir (por ejemplo) 64 coeficientes por frame
model = AstronomicalDistortionEstimator(
    cnn_channels=32,
    spatial_dim=128,
    num_coefficients=64, # N coeficientes a elegir
    freeze_cnn=True      # Congelado recomendado para mitigar "few-shot"
)

# Simulamos un lote con 2 stacks rellenos a max_len = 500
# Dimensiones: [Batch_Size=2, Max_Frames=500, H=128, W=128]
batch_images = torch.randn(2, 500, 128, 128) 

# Creamos la máscara para un lote heterogéneo:
# Stack 0: 494 frames reales + 6 de padding (los últimos 6 son True)
# Stack 1: 500 frames reales (todos son False)
padding_mask = torch.zeros((2, 500), dtype=torch.bool)
padding_mask[0, 494:] = True # Marcamos los 6 frames falsos

# --- FORWARD PASS COMPLETO ---
predicted_coefficients = model(batch_images, padding_mask=padding_mask)

print("Entrada al modelo:", batch_images.shape)
print("Salida de coeficientes:", predicted_coefficients.shape) 
# Salida: torch.Size([2, 500, 64])

# --- USO EN TU PÉRDIDA FÍSICA ---
# Para el stack de 494 frames, simplemente extraes la porción válida:
valid_coefs_stack_0 = predicted_coefficients[0, :494, :] # [494, 64]
valid_coefs_stack_1 = predicted_coefficients[1, :500, :] # [500, 64]