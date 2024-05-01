from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import math

# Erstellen einer SQLite-Datenbank und einer Verbindung dazu
engine = create_engine('sqlite:///training_database.db')

# Laden der CSV-Dateien
train_df = pd.read_csv('train.csv')
ideal_df = pd.read_csv('ideal.csv')
test_df = pd.read_csv('test.csv')

# Importieren der Trainingsdaten in die Tabelle 'training_data'
train_df.to_sql('training_data', engine, index=False, if_exists='replace')

# Importieren der idealen Funktionen in die Tabelle 'ideal_functions'
ideal_df.to_sql('ideal_functions', engine, index=False, if_exists='replace')

# Importieren der Testdaten in die Tabelle 'test_data'
test_df.to_sql('test_data', engine, index=False, if_exists='replace')



# Funktion zur Berechnung der Summe der quadratischen Abweichungen
def calculate_least_squares(train_y, ideal_y):
    return np.sum((train_y - ideal_y) ** 2)

# Beste Fits für jeden Trainingsdatensatz finden
best_fits = {}
for column in train_df.columns[1:]:  # Ignoriere die erste Spalte (x-Werte)
    best_fit = None
    min_error = float('inf')
    for ideal_column in ideal_df.columns[1:]:  # Ignoriere die erste Spalte (x-Werte)
        error = calculate_least_squares(train_df[column], ideal_df[ideal_column])
        if error < min_error:
            min_error = error
            best_fit = ideal_column
    best_fits[column] = best_fit

best_fits


# Funktion zur Berechnung der maximalen Abweichung
def calculate_max_deviation(train_y, ideal_y):
    return np.max(np.abs(train_y - ideal_y))

# Berechnen der maximalen Abweichung für jede ideale Funktion und die Trainingsdaten
max_deviations = {key: calculate_max_deviation(train_df[key], ideal_df[best_fits[key]]) 
                  for key in best_fits}

# Validierung mit Testdaten
validations = []
for index, row in test_df.iterrows():
    x_value = row['x']
    y_value = row['y']
    for train_column, ideal_column in best_fits.items():
        ideal_y_value = ideal_df.loc[ideal_df['x'] == x_value, ideal_column].iloc[0]
        deviation = abs(y_value - ideal_y_value)
        if deviation <= max_deviations[train_column] * math.sqrt(2):
            validations.append({
                'x': x_value,
                'y': y_value,
                'train_column': train_column,
                'ideal_function': ideal_column,
                'deviation': deviation
            })

# Konvertieren der Validierungsergebnisse in einen DataFrame
validation_df = pd.DataFrame(validations)

validation_df

import matplotlib.pyplot as plt

# Auswahl einer Spalte aus den Trainingsdaten für die Visualisierung.

train_column = train_df.columns[1]
ideal_column = best_fits[train_column]

# Erstellen der Visualisierung
plt.figure(figsize=(10, 6))

# Trainingsdaten plotten
plt.scatter(train_df['x'], train_df[train_column], label='Trainingsdaten', color='blue', linewidth=2)

# Ideale Funktion plotten
plt.plot(ideal_df['x'], ideal_df[ideal_column], label='Ideale Funktion', color='red', linewidth=2)

# Testdaten plotten
plt.scatter(test_df['x'], test_df['y'], label='Testdaten', color='green', marker='x')

# Beschriftungen und Legende hinzufügen
plt.xlabel('x')
plt.ylabel('y')
plt.title('Visualisierung der Trainingsdaten, idealen Funktion und Testdaten')
plt.legend()

# Zeige die Visualisierung
plt.show()

# Erstellen von Diagrammen für jede Paarung von Trainingsdaten und ihrer besten Übereinstimmung
fig, axs = plt.subplots(len(best_fits), 1, figsize=(10, 6 * len(best_fits)))

if len(best_fits) == 1:  
    axs = [axs]

for idx, (train_col, ideal_col) in enumerate(best_fits.items()):
    axs[idx].scatter(train_df['x'], train_df[train_col], label=f'Trainingsdaten ({train_col})', alpha=0.5)
    axs[idx].plot(ideal_df['x'], ideal_df[ideal_col], color='red', label=f'Idealfunktion ({ideal_col})')
    axs[idx].set_title(f'Vergleich von Trainingsdaten ({train_col}) mit Idealfunktion ({ideal_col})')
    axs[idx].set_xlabel('x')
    axs[idx].set_ylabel('y')
    axs[idx].legend()
    axs[idx].grid(True)
 

plt.tight_layout()
plt.show()

