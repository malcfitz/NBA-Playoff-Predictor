import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from FetchData import FEATURES


def train_model():

    df = pd.read_csv(
        'data/playoff_training_data.csv'
    )

    df = df.dropna()

    X = df[FEATURES]
    y = df['RESULT']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        C=0.5
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nModel Accuracy: {accuracy:.2%}")

    print("\nFeature Importance:")

    for feature, coef in zip(FEATURES, model.coef_[0]):
        print(f"{feature:<20} {coef:.3f}")

    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("\nModel and scaler saved.")


if __name__ == "__main__":
    train_model()