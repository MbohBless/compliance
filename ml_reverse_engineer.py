import json
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

class LegacySystemReverseEngineer:
    def __init__(self):
        self.models = {}
        self.feature_engineers = {}
        self.best_model = None
        self.best_score = float('inf')

    def load_data(self):
        """Load and prepare the training data"""
        with open('public_cases.json', 'r') as f:
            data = json.load(f)

        features = []
        targets = []

        for case in data:
            inp = case['input']
            features.append([
                inp['trip_duration_days'],
                inp['miles_traveled'],
                inp['total_receipts_amount']
            ])
            targets.append(case['expected_output'])

        self.X = np.array(features)
        self.y = np.array(targets)
        self.df = pd.DataFrame(self.X, columns=['days', 'miles', 'receipts'])
        self.df['target'] = self.y

        print(f"Loaded {len(self.X)} training examples")
        print(f"Feature ranges: Days {self.X[:,0].min()}-{self.X[:,0].max()}, "
              f"Miles {self.X[:,1].min()}-{self.X[:,1].max()}, "
              f"Receipts ${self.X[:,2].min():.2f}-${self.X[:,2].max():.2f}")
        return self.X, self.y

    def engineer_features(self, X):
        """Create sophisticated feature engineering"""
        features = X.copy()
        days, miles, receipts = X[:, 0], X[:, 1], X[:, 2]

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(X)

        engineered = np.column_stack([
            features,
            miles / np.maximum(days, 1),
            receipts / np.maximum(days, 1),
            receipts / np.maximum(miles, 1),
            days * miles,
            days * receipts,
            miles * receipts,
            np.log1p(days),
            np.log1p(miles),
            np.log1p(receipts),
            days ** 2,
            miles ** 2,
            receipts ** 2,
            (days == 1).astype(int),
            (days >= 7).astype(int),
            (days >= 14).astype(int),
            (receipts < 100).astype(int),
            (receipts > 1000).astype(int),
            (receipts > 2000).astype(int),
            (miles < 100).astype(int),
            (miles > 500).astype(int),
            (miles > 1000).astype(int),
        ])

        return engineered

    def train_multiple_models(self):
        """Train multiple ML models to find the best approach"""
        X_eng = self.engineer_features(self.X)

        models_to_try = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic': ElasticNet(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

        print("Training multiple ML models...")
        for name, model in models_to_try.items():
            try:
                cv_scores = cross_val_score(
                    model, X_eng, self.y,
                    cv=5, scoring='neg_mean_absolute_error'
                )
                avg_score = -cv_scores.mean()

                model.fit(X_eng, self.y)
                predictions = model.predict(X_eng)
                mae = mean_absolute_error(self.y, predictions)

                self.models[name] = {
                    'model': model,
                    'cv_score': avg_score,
                    'mae': mae
                }

                print(f"{name:>8}: CV MAE = ${avg_score:.2f}, Train MAE = ${mae:.2f}")

                if avg_score < self.best_score:
                    self.best_score = avg_score
                    self.best_model = model

            except Exception as e:
                print(f"Failed to train {name}: {e}")

    def optimize_custom_formula(self):
        """Use scipy optimization to find custom formula parameters"""
        print("\nOptimizing custom formula...")

        def custom_formula(params, X):
            days, miles, receipts = X[:, 0], X[:, 1], X[:, 2]
            day_rate, mile_rate = params[0], params[1]
            receipt_rate1, receipt_thresh1 = params[2], params[3]
            receipt_rate2, receipt_thresh2 = params[4], params[5]
            receipt_rate3 = params[6]

            day_corrections = {}
            for i, day_len in enumerate(range(1, 13)):
                if i + 7 < len(params):
                    day_corrections[day_len] = params[i + 7]
                else:
                    day_corrections[day_len] = 1.0

            base = days * day_rate + miles * mile_rate
            receipt_contrib = np.zeros_like(receipts)

            tier1 = np.minimum(receipts, receipt_thresh1)
            receipt_contrib += tier1 * receipt_rate1

            remaining = np.maximum(receipts - receipt_thresh1, 0)
            tier2 = np.minimum(remaining, receipt_thresh2 - receipt_thresh1)
            receipt_contrib += tier2 * receipt_rate2

            tier3 = np.maximum(receipts - receipt_thresh2, 0)
            receipt_contrib += tier3 * receipt_rate3

            total = base + receipt_contrib
            for day_len, correction in day_corrections.items():
                mask = (days == day_len)
                total[mask] *= correction

            return total

        def objective(params):
            try:
                predictions = custom_formula(params, self.X)
                return np.mean(np.abs(predictions - self.y))
            except:
                return 1e6

        initial_params = [
            80.0, 0.5, 0.6, 1000, 0.3, 2000, 0.1
        ] + [1.0] * 12

        bounds = [
            (50, 120), (0.1, 1.0), (0.1, 1.0), (500, 1500),
            (0.05, 0.8), (1500, 3000), (0.01, 0.5),
        ] + [(0.5, 1.5)] * 12

        result = optimize.minimize(
            objective, initial_params, bounds=bounds, method='L-BFGS-B'
        )

        if result.success:
            optimized_mae = result.fun
            print(f"Optimized formula MAE: ${optimized_mae:.2f}")
            self.optimized_params = result.x
            self.optimized_formula = lambda X: custom_formula(result.x, X)

            if optimized_mae < self.best_score:
                self.best_score = optimized_mae
                self.best_model = 'optimized_formula'

            return result.x
        else:
            print("Optimization failed")
            return initial_params

    def analyze_patterns(self):
        """Analyze data patterns to understand the legacy system"""
        print("\nAnalyzing data patterns...")

        day_groups = self.df.groupby('days')['target'].agg(['mean', 'std', 'count'])
        print("\nReimbursement by trip duration:")
        for days, stats in day_groups.iterrows():
            if stats['count'] >= 5:
                print(f"  {int(days):2d} days: ${stats['mean']:7.2f} ± "
                      f"${stats['std']:6.2f} ({int(stats['count'])} cases)")

        receipt_bins = [0, 100, 500, 1000, 1500, 2000, 3000, 10000]
        self.df['receipt_bin'] = pd.cut(self.df['receipts'], bins=receipt_bins)
        receipt_groups = self.df.groupby('receipt_bin')['target'].agg(['mean', 'std', 'count'])
        print("\nReimbursement by receipt amount:")
        for bin_range, stats in receipt_groups.iterrows():
            if stats['count'] >= 5:
                print(f"  {str(bin_range):15s}: ${stats['mean']:7.2f} ± "
                      f"${stats['std']:6.2f} ({int(stats['count'])} cases)")

        return {}

    def predict(self, days, miles, receipts):
        """Make prediction using the best model"""
        X_test = np.array([[days, miles, receipts]])
        if self.best_model == 'optimized_formula':
            return float(self.optimized_formula(X_test)[0])
        else:
            X_test_eng = self.engineer_features(X_test)
            return float(self.best_model.predict(X_test_eng)[0])

    def save_model_summary(self):
        """Save a summary of the best model for implementation"""
        name = (type(self.best_model).__name__
                if hasattr(self.best_model, '__name__')
                else self.best_model)
        print(f"\nBest model: {name}")
        print(f"Best score: ${self.best_score:.2f}")

        if hasattr(self, 'optimized_params'):
            print("\nOptimized formula parameters:")
            param_names = [
                'day_rate', 'mile_rate', 'receipt_rate1', 'receipt_thresh1',
                'receipt_rate2', 'receipt_thresh2', 'receipt_rate3'
            ] + [f'day_{i}_correction' for i in range(1, 13)]
            for name, value in zip(param_names, self.optimized_params):
                print(f"  {name:20s}: {value:.4f}")

def main():
    """Main training and optimization loop"""
    print("ML-Based Legacy System Reverse Engineering")
    print("=" * 60)

    engineer = LegacySystemReverseEngineer()
    X, y = engineer.load_data()
    engineer.analyze_patterns()
    engineer.train_multiple_models()
    engineer.optimize_custom_formula()
    engineer.save_model_summary()

    test_cases = [
        (3, 93, 1.42),
        (5, 250, 150.75),
        (1, 55, 3.6),
        (12, 800, 2000)
    ]
    for days, miles, receipts in test_cases:
        prediction = engineer.predict(days, miles, receipts)
        print(f"  {days}d, {miles}mi, ${receipts:.2f} → ${prediction:.2f}")

    return engineer

if __name__ == "__main__":
    main()
