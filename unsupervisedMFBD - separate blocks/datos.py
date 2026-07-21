import pandas as pd

# Creamos una tabla rápida
data = {
    'Nombre': ['Paula', 'Vaca', 'Python'],
    'Nivel': [100, 50, 99]
}

df = pd.DataFrame(data)

print("¡Tabla creada con éxito!")
print(df)