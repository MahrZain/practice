import joblib
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load model if it exists, else create a new model
model_path = 'book_rating_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = LinearRegression()

# Loop over the pages of the website
cleaned_books = []
for i in range(10, 100):  # Loop over pages 1 to 20
    url = f'http://books.toscrape.com/catalogue/page-{i}.html'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    books = soup.find_all('article', class_='product_pod')

    for book in books:
        title = book.h3.a['title']
        price = book.find('p', class_='price_color').text
        rating = book.p['class'][1]  # Rating class: 'One', 'Two', etc.

        # Convert ratings to numerical values
        rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
        rating_value = rating_map.get(rating, 0)  # Default to 0 if no rating found

        cleaned_books.append({
            "title": title.strip(),
            "price": float(price.strip('Â£')),  # Remove currency symbol
            "rating": rating_value
        })

# Convert the cleaned_books list into a DataFrame for easier manipulation
df = pd.DataFrame(cleaned_books)

# Features (independent variables)
X = df[['price']]  # Using price as the feature for simplicity

# Target (dependent variable)
y = df['rating']  # We're predicting the rating (1 to 5) based on price

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train or fine-tune the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the updated model
joblib.dump(model, model_path)
