# Hotel Booking Cancellation Prediction

A complete machine learning project to predict whether a customer will cancel a hotel reservation or not.

## What it does

- Predicts booking cancellation based on features like lead time, number of adults/children, meal plan, market segment, average price per room, etc.
- Compares 3 feature scaling techniques: StandardScaler, MinMaxScaler, RobustScaler
- Compares 5 outlier detection methods: Z-Score, IQR, Isolation Forest, LOF (Local Outlier Factor), Elliptic Envelope
- Trains and compares 10 machine learning models across each outlier-detection method:
  - Logistic Regression
  - KNN
  - Naive Bayes
  - SVM
  - Decision Tree
  - Random Forest
  - Extra Trees
  - Bagging
  - AdaBoost
  - Gradient Boosting
- Performs hyperparameter tuning using GridSearchCV for each model/outlier-method combination

## Best Result

Elliptic Envelope + Logistic Regression achieved the highest accuracy among all tested combinations, at **67.4%**.

## Known Limitation

Some of the outlier-detection methods (particularly LOF and Elliptic Envelope) removed a very large portion of the dataset before the train/test split, leaving only a few hundred rows for evaluation instead of the original ~36,000. This is a known issue with the outlier-removal thresholds used and is a planned area for revisiting — a less aggressive filtering approach should allow the models to train and be evaluated on a much larger, more representative portion of the data.

## Technologies

- Python
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn

## How to Run

1. Clone the repository
2. Install requirements: pip install -r requirements.txt
3. Place Hotel Reservation.csv in the same directory
4. Run the notebook in Jupyter or Google Colab

## Dataset

Hotel Reservation.csv with features including:
- lead_time, no_of_adults, no_of_children, no_of_special_requests
- type_of_meal_plan, room_type_reserved, market_segment_type
- avg_price_per_room, arrival_date, arrival_month, arrival_year
- repeated_guest, booking_status (target)

Original dataset size: 36,275 rows.

## Author
**Fatma Abdullah**
