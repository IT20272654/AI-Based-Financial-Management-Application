# AI-Based-Financial-Management-Application
The AI Financial Management System is a project designed to simplify and enhance personal financial management. Using Artificial Intelligence, it helps users with financial planning, investment management, debt management, and savings management. By providing predictive analytics, and personalized recommendations, this system empowers users to make smarter financial decisions and achieve their financial goals effectively.

This is the AI/ML component of a financial management application, developed using Google Colab. The project allows users to scan their receipts or bills, which are processed using Tesseract OCR to extract key financial details like date and amount. Users can also manually add financial records. The system then organizes this data, enables financial data visualization, and provides a 7-day prediction of future financial trends using machine learning models.

---

##  Main Libraries Used

###  Image Processing & OCR
- **Tesseract OCR (`pytesseract`)** – Extracts text from scanned receipts/bills.
- **OpenCV (`cv2`)** – Handles image preprocessing (e.g., grayscale, thresholding).
- **Pillow (`PIL`)** – Image handling and conversion.

###  Data Analysis & Visualization
- **Pandas** – Handles financial data frames and preprocessing.
- **NumPy** – Numerical computations.
- **Matplotlib** – Plots and visualizes data trends.
- **Seaborn** – Enhances data visualizations with statistical plots.

###  Machine Learning
- **XGBoost** – Builds a model for 7-day financial forecasting.
- **Scikit-learn (`sklearn.metrics`)** – Measures prediction accuracy (MSE, MAE, R²).

### 🛠 Utility
- **Google Colab `files` module** – For uploading images or CSVs in Colab.
- **Datetime** – For date handling and formatting.
- **Regex (`re`)** – Extracts dates and amounts from text.
- **Warnings** – Manages and suppresses warning messages.

---

##  Features

- Scan bills/receipts and extract date and amount.
- Add financial data manually or via OCR.
- Visualize financial data using graphs.
- Predict financial trends for the next 7 days using ML.




