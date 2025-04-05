from flask import Flask, render_template, request, jsonify
import joblib  # Pour charger le modèle
import numpy as np  # Pour manipuler les données

# Initialisation de l'application Flask
app = Flask(__name__ ,template_folder="templates1")

# Charger le modèle de prédiction
try:
    model = joblib.load('diabetes_model.pkl')  # Chemin vers votre modèle sauvegardé
except FileNotFoundError:
    raise Exception("Le modèle 'diabetes_model.pkl' n'a pas été trouvé. Veuillez vérifier le chemin du fichier.")

@app.route('/')
def home():
    """Afficher la page d'accueil avec le formulaire"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prédire si une personne est diabétique ou non"""
    try:
        # Récupérer les données du formulaire
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = int(request.form['age'])

        # Créer un tableau avec les données utilisateur
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

        # Effectuer la prédiction avec le modèle
        prediction = model.predict(input_data)

        if prediction[0] == 0:

          result = 'Vous êtes malade (diabète).'
        else: 
            result='Vous n\'êtes pas malade (pas de diabète).'  
        # Affichage du résultat et log
        print("Affichage du résultat:", result) 

        return render_template('result.html', result=result)

    except Exception as e:
        # Renvoyer un message d'erreur détaillé si une exception se produit
        return jsonify({'error': f"Une erreur est survenue : {str(e)}"})

if __name__ == "__main__":
  
    app.run(debug=True)
