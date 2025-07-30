# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:59:58 2025

@author: crist
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from scipy import stats

# Cargar datos CSV con separador ;
df = pd.read_csv("precio_mes_combustible_20250730.csv", sep=";")

# Eliminar columnas innecesarias
cols_to_drop = ["Codigo_departamento", "Codigo_municipio", "Direccion", "Nombre_comercial"]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Guardar dataset limpio
df.to_csv("dataset_combustible_limpio.csv", index=False)

# INFORMACIÓN INICIAL DEL DATASET
# Ubicación en informe: Sección "Metodología" - Subsección "Descripción de los datos"
print("=== INFORMACIÓN INICIAL DEL DATASET ===")
print(f"Dimensiones del dataset: {df.shape}")
print(f"Columnas disponibles: {list(df.columns)}")
print("\nPrimeras 5 filas del dataset:")
print(df.head())
print("\nInformación general del dataset:")
print(df.info())

# ------------------ Visualizaciones Mejoradas ------------------

# IMAGEN 1: Análisis Temporal de Precios
# Ubicación en informe: Sección "Análisis Temporal" - Subsección "Evolución de precios por mes"
plt.figure(figsize=(14, 8))
# Crear gráfico de líneas por tipo de combustible a lo largo del tiempo
for producto in df['Producto'].unique():
    data_producto = df[df['Producto'] == producto]
    precio_mes = data_producto.groupby('Mes')['Precio'].mean()
    plt.plot(precio_mes.index, precio_mes.values, marker='o', linewidth=2, label=producto)

plt.title("Evolución de Precios por Tipo de Combustible a lo Largo del Tiempo", fontsize=16, fontweight='bold')
plt.xlabel("Mes", fontsize=12)
plt.ylabel("Precio Promedio (COP)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("evolucion_precios_temporal.png", dpi=300, bbox_inches='tight')
plt.show()

# IMAGEN 2: Análisis Comparativo por Marca/Bandera
# Ubicación en informe: Sección "Análisis por Marca" - Subsección "Comparación de precios entre marcas"
plt.figure(figsize=(14, 8))
# Boxplot mejorado con información estadística
sns.boxplot(data=df, x="Bandera", y="Precio", palette="Set3")
plt.title("Distribución de Precios por Marca de Estación de Servicio", fontsize=16, fontweight='bold')
plt.xlabel("Marca/Bandera", fontsize=12)
plt.ylabel("Precio (COP)", fontsize=12)
plt.xticks(rotation=45)
# Agregar línea de precio promedio general
plt.axhline(y=df['Precio'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Promedio General: {df["Precio"].mean():.0f} COP')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("boxplot_precios_por_marca.png", dpi=300, bbox_inches='tight')
plt.show()

# IMAGEN 3: Histograma Mejorado con Estadísticas
# Ubicación en informe: Sección "Análisis Estadístico Descriptivo" - Subsección "Distribución de frecuencias"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histograma principal
sns.histplot(df["Precio"], bins=25, kde=True, ax=ax1, color='skyblue', alpha=0.7)
ax1.axvline(df['Precio'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["Precio"].mean():.0f}')
ax1.axvline(df['Precio'].median(), color='green', linestyle='-', linewidth=2, label=f'Mediana: {df["Precio"].median():.0f}')
ax1.set_title("Distribución de Precios de Combustibles", fontsize=14, fontweight='bold')
ax1.set_xlabel("Precio (COP)", fontsize=12)
ax1.set_ylabel("Frecuencia", fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# QQ Plot para normalidad
from scipy import stats
stats.probplot(df['Precio'], dist="norm", plot=ax2)
ax2.set_title("Q-Q Plot: Evaluación de Normalidad", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("histograma_avanzado_precios.png", dpi=300, bbox_inches='tight')
plt.show()

# IMAGEN 4: Análisis Detallado por Tipo de Combustible
# Ubicación en informe: Sección "Análisis por Tipo de Producto" - Subsección "Comparación exhaustiva entre combustibles"
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Boxplot detallado
sns.boxplot(data=df, x="Producto", y="Precio", ax=ax1, palette="viridis")
ax1.set_title("Distribución de Precios por Tipo de Combustible", fontsize=14, fontweight='bold')
ax1.set_xlabel("Tipo de Combustible", fontsize=11)
ax1.set_ylabel("Precio (COP)", fontsize=11)
ax1.tick_params(axis='x', rotation=45, labelsize=9)
ax1.grid(True, alpha=0.3)

# Violin plot para mostrar densidad
sns.violinplot(data=df, x="Producto", y="Precio", ax=ax2, palette="plasma")
ax2.set_title("Densidad de Distribución por Combustible", fontsize=14, fontweight='bold')
ax2.set_xlabel("Tipo de Combustible", fontsize=11)
ax2.set_ylabel("Precio (COP)", fontsize=11)
ax2.tick_params(axis='x', rotation=45, labelsize=9)
ax2.grid(True, alpha=0.3)

# Gráfico de barras con estadísticas
precio_stats = df.groupby('Producto')['Precio'].agg(['mean', 'std', 'min', 'max'])
precio_stats['mean'].plot(kind='bar', ax=ax3, color='lightcoral', alpha=0.8)
ax3.set_title("Precio Promedio por Tipo de Combustible", fontsize=14, fontweight='bold')
ax3.set_xlabel("Tipo de Combustible", fontsize=11)
ax3.set_ylabel("Precio Promedio (COP)", fontsize=11)
ax3.tick_params(axis='x', rotation=45, labelsize=9)
ax3.grid(True, alpha=0.3)

# Gráfico de participación mejorado
counts = df["Producto"].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
wedges, texts, autotexts = ax4.pie(counts.values, labels=counts.index, autopct='%1.1f%%', 
                                   startangle=90, colors=colors, textprops={'fontsize': 9})
ax4.set_title("Participación de Mercado por Combustible", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("analisis_detallado_combustibles.png", dpi=300, bbox_inches='tight')
plt.show()

# IMAGEN 5: Matriz de Correlación y Análisis de Variables
# Ubicación en informe: Sección "Análisis de Correlaciones" - Subsección "Relaciones entre variables"
plt.figure(figsize=(12, 8))

# Crear matriz de correlación con variables numéricas y categóricas codificadas
df_corr = pd.get_dummies(df[['Periodo', 'Mes', 'Bandera', 'Producto', 'Precio']], drop_first=True)
correlation_matrix = df_corr.corr()

# Heatmap mejorado
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
plt.title("Matriz de Correlación entre Variables", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("matriz_correlacion.png", dpi=300, bbox_inches='tight')
plt.show()

# IMAGEN 6: Análisis de Outliers Detallado
# Ubicación en informe: Sección "Detección de Anomalías" - Subsección "Identificación de valores atípicos"
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Boxplot general para outliers
sns.boxplot(y=df['Precio'], ax=ax1, color='lightblue')
ax1.set_title("Detección General de Outliers", fontsize=14, fontweight='bold')
ax1.set_ylabel("Precio (COP)", fontsize=12)
ax1.grid(True, alpha=0.3)

# Scatter plot con outliers marcados
Q1 = df['Precio'].quantile(0.25)
Q3 = df['Precio'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Precio'] < lower_bound) | (df['Precio'] > upper_bound)]
normal_data = df[(df['Precio'] >= lower_bound) & (df['Precio'] <= upper_bound)]

ax2.scatter(range(len(normal_data)), normal_data['Precio'], color='blue', alpha=0.6, label='Datos Normales')
ax2.scatter(outliers.index, outliers['Precio'], color='red', alpha=0.8, label=f'Outliers ({len(outliers)})')
ax2.set_title("Identificación de Outliers en el Dataset", fontsize=14, fontweight='bold')
ax2.set_xlabel("Índice de Observación", fontsize=12)
ax2.set_ylabel("Precio (COP)", fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Histograma con outliers marcados
ax3.hist(normal_data['Precio'], bins=20, alpha=0.7, color='blue', label='Datos Normales')
ax3.hist(outliers['Precio'], bins=10, alpha=0.8, color='red', label='Outliers')
ax3.set_title("Distribución: Normales vs Outliers", fontsize=14, fontweight='bold')
ax3.set_xlabel("Precio (COP)", fontsize=12)
ax3.set_ylabel("Frecuencia", fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analisis_outliers_detallado.png", dpi=300, bbox_inches='tight')
plt.show()

# ------------------ Modelo Predictivo Mejorado ------------------
# RESULTADO DE MODELO: Predicción de Precio Futuro
# Ubicación en informe: Sección "Modelo Predictivo" - Subsección "Estimación de precios futuros"

print("\n" + "="*60)
print("ANÁLISIS PREDICTIVO AVANZADO")
print("="*60)

# Codificar variables categóricas
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(columns=["Precio"])
y = df_encoded["Precio"]

# División de datos para validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo de regresión lineal
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 MÉTRICAS DEL MODELO:")
print(f"   • R² Score: {r2:.4f} ({r2*100:.2f}% de varianza explicada)")
print(f"   • Error Cuadrático Medio (MSE): {mse:,.2f}")
print(f"   • Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} COP")
print(f"   • Error Absoluto Medio (MAE): {mae:.2f} COP")

# Predicción futura
future_input = np.mean(X_train_scaled, axis=0).reshape(1, -1)
future_price = model.predict(future_input)[0]
print(f"\n🔮 PREDICCIÓN FUTURA:")
print(f"   • Precio promedio estimado: {future_price:.2f} COP")

# Análisis de características importantes
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False).head(10)

print(f"\n📈 TOP 10 VARIABLES MÁS INFLUYENTES:")
for idx, row in feature_importance.iterrows():
    print(f"   • {row['Feature']}: {row['Coefficient']:+.2f}")

# IMAGEN 7: Evaluación del Modelo Predictivo
# Ubicación en informe: Sección "Modelo Predictivo" - Subsección "Validación del modelo"
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico de predicción vs realidad
ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Precios Reales', fontsize=12)
ax1.set_ylabel('Precios Predichos', fontsize=12)
ax1.set_title(f'Predicción vs Realidad (R² = {r2:.3f})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Residuos
residuos = y_test - y_pred
ax2.scatter(y_pred, residuos, alpha=0.6, color='green')
ax2.axhline(y=0, color='red', linestyle='--')
ax2.set_xlabel('Predicciones', fontsize=12)
ax2.set_ylabel('Residuos', fontsize=12)
ax2.set_title('Análisis de Residuos', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Distribución de residuos
ax3.hist(residuos, bins=20, alpha=0.7, color='orange', edgecolor='black')
ax3.set_xlabel('Residuos', fontsize=12)
ax3.set_ylabel('Frecuencia', fontsize=12)
ax3.set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Importancia de características
top_features = feature_importance.head(8)
bars = ax4.barh(range(len(top_features)), top_features['Abs_Coefficient'], color='purple', alpha=0.7)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['Feature'], fontsize=10)
ax4.set_xlabel('Importancia (Coeficiente Absoluto)', fontsize=12)
ax4.set_title('Variables Más Influyentes en el Precio', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("evaluacion_modelo_predictivo.png", dpi=300, bbox_inches='tight')
plt.show()

# ESTADÍSTICAS AVANZADAS PARA EL INFORME
# Ubicación en informe: Sección "Resumen Ejecutivo" y "Estadísticas Descriptivas"
print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS AVANZADAS")
print("="*60)

# Estadísticas generales
print(f"📊 ESTADÍSTICAS GENERALES:")
print(f"   • Precio promedio general: {df['Precio'].mean():.2f} COP")
print(f"   • Precio mediano: {df['Precio'].median():.2f} COP")
print(f"   • Precio mínimo: {df['Precio'].min():.2f} COP")
print(f"   • Precio máximo: {df['Precio'].max():.2f} COP")
print(f"   • Desviación estándar: {df['Precio'].std():.2f} COP")
print(f"   • Coeficiente de variación: {(df['Precio'].std()/df['Precio'].mean())*100:.2f}%")
print(f"   • Rango de precios: {df['Precio'].max() - df['Precio'].min():.2f} COP")

# Información del dataset
print(f"\n🗂️  INFORMACIÓN DEL DATASET:")
print(f"   • Total de observaciones: {len(df):,}")
print(f"   • Número de municipios: {df['Municipio'].nunique()}")
print(f"   • Número de marcas/banderas: {df['Bandera'].nunique()}")
print(f"   • Tipos de combustible: {df['Producto'].nunique()}")

# Análisis por tipo de combustible
print(f"\n⛽ ANÁLISIS POR TIPO DE COMBUSTIBLE:")
for producto in df['Producto'].unique():
    data_producto = df[df['Producto'] == producto]
    print(f"   • {producto}:")
    print(f"     - Precio promedio: {data_producto['Precio'].mean():.2f} COP")
    print(f"     - Observaciones: {len(data_producto)} ({len(data_producto)/len(df)*100:.1f}%)")
    print(f"     - Rango: {data_producto['Precio'].min():.0f} - {data_producto['Precio'].max():.0f} COP")

# Análisis por marca
print(f"\n🏢 ANÁLISIS POR MARCA/BANDERA:")
marca_stats = df.groupby('Bandera')['Precio'].agg(['count', 'mean', 'std', 'min', 'max'])
for marca in marca_stats.index:
    stats = marca_stats.loc[marca]
    print(f"   • {marca}:")
    print(f"     - Precio promedio: {stats['mean']:.2f} COP")
    print(f"     - Estaciones: {stats['count']} ({stats['count']/len(df)*100:.1f}%)")
    print(f"     - Desv. estándar: {stats['std']:.2f} COP")

# Detección de outliers
Q1 = df['Precio'].quantile(0.25)
Q3 = df['Precio'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Precio'] < Q1 - 1.5*IQR) | (df['Precio'] > Q3 + 1.5*IQR)]

print(f"\n🚨 DETECCIÓN DE ANOMALÍAS:")
print(f"   • Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print(f"   • Q1 (25%): {Q1:.2f} COP")
print(f"   • Q3 (75%): {Q3:.2f} COP")
print(f"   • Rango intercuartílico (IQR): {IQR:.2f} COP")

if len(outliers) > 0:
    print(f"   • Precio outlier mínimo: {outliers['Precio'].min():.2f} COP")
    print(f"   • Precio outlier máximo: {outliers['Precio'].max():.2f} COP")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO - IMÁGENES GENERADAS")
print("="*60)
