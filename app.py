from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, r2_score
import os

app = Flask(__name__)


class WheatYieldPredictor:
    def __init__(self):
        self.regression_models = {}
        self.classification_models = {}
        self.best_reg_pipe = None
        self.best_clf_pipe = None
        self.prep_reg = None
        self.prep_clf = None
        self.is_trained = False

    def train_models(self, df_path="data/crop_yield.csv"):
        """Train both regression and classification models"""
        if isinstance(df_path, str):
            if not os.path.exists(df_path):
                raise FileNotFoundError(f"Dataset file not found: {df_path}")
            df = pd.read_csv(df_path)
        else:
            df = df_path

        #  Keep only wheat rows
        df = df[df['Crop'].str.lower() == 'wheat'].copy()

        if df.empty:
            raise ValueError("No rows with Crop == 'Wheat' found in dataset!")

        # Drop Crop column since it's always Wheat now
        df = df.drop(columns=['Crop'])

        # === Features (X) and Target (y) ===
        X = df.drop('Yield', axis=1)
        y = df['Yield']

        # === Categorical features ===
        cat_feats = ['Season', 'State']

        # === REGRESSION ===
        reg_models = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.prep_reg = ColumnTransformer(
            [('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)],
            remainder='passthrough'
        )

        best_r2 = -float('inf')
        best_reg_name = None

        for name, model in reg_models.items():
            pipe = Pipeline([('prep', self.prep_reg), ('model', model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            r2 = r2_score(y_test, pred)

            if r2 > best_r2:
                best_r2 = r2
                best_reg_name = name
                self.best_reg_pipe = pipe

        # === CLASSIFICATION ===
        y_bins = pd.qcut(y, q=3, labels=['Low', 'Medium', 'High'])

        clf_models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        }

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_bins, test_size=0.2, random_state=42, stratify=y_bins
        )

        self.prep_clf = ColumnTransformer(
            [('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)],
            remainder='passthrough'
        )

        best_f1 = -float('inf')
        best_clf_name = None

        for name, model in clf_models.items():
            pipe = Pipeline([('prep', self.prep_clf), ('model', model)])
            pipe.fit(Xc_train, yc_train)
            pred = pipe.predict(Xc_test)
            report = classification_report(yc_test, pred, output_dict=True, zero_division=0)
            f1_macro = report['macro avg']['f1-score']

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_clf_name = name
                self.best_clf_pipe = pipe

        self.is_trained = True
        print(f"Models trained successfully!")
        print(f"Best Regression Model: {best_reg_name} (R²: {best_r2:.3f})")
        print(f"Best Classification Model: {best_clf_name} (F1: {best_f1:.3f})")

        return df

    def predict(self, sample_data):
        """Make predictions for new data"""
        if not self.is_trained:
            raise ValueError("Models not trained yet!")

        sample_df = pd.DataFrame([sample_data])

        yield_pred = self.best_reg_pipe.predict(sample_df)[0]
        class_pred = self.best_clf_pipe.predict(sample_df)[0]

        # Get prediction probabilities for classification
        class_probs = self.best_clf_pipe.predict_proba(sample_df)[0]
        class_names = self.best_clf_pipe.named_steps['model'].classes_

        prob_dict = {class_names[i]: float(class_probs[i]) for i in range(len(class_names))}

        return {
            'yield_prediction': float(yield_pred),
            'yield_class': str(class_pred),
            'class_probabilities': prob_dict
        }


# Initialize the predictor for console use
predictor = WheatYieldPredictor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train_models():
    try:
        sample_data = predictor.train_models("data/crop_yield.csv")
        return jsonify({
            'success': True,
            'message': 'Models trained successfully!',
            'data_shape': sample_data.shape,
            'sample_stats': {
                'mean_yield': float(sample_data['Yield'].mean()),
                'std_yield': float(sample_data['Yield'].std()),
                'min_yield': float(sample_data['Yield'].min()),
                'max_yield': float(sample_data['Yield'].max())
            }
        })


    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate required fields
        required_fields = ['Production', 'Area', 'Crop_Year', 'Annual_Rainfall',
                           'Fertilizer', 'Pesticide', 'Season', 'State']

        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'})

        # Convert numeric fields
        numeric_fields = ['Production', 'Area', 'Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        for field in numeric_fields:
            try:
                data[field] = float(data[field])
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': f'Invalid numeric value for {field}'})

        # Make prediction
        result = predictor.predict(data)
        result['success'] = True

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_options')
def get_options():
    try:
        df = pd.read_csv("data/crop_yield.csv")
        df = df[df['Crop'].str.lower() == 'wheat']

        states = sorted(df['State'].unique().tolist())
        seasons = sorted(df['Season'].unique().tolist())

        return jsonify({
            'states': states,
            'seasons': seasons
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)