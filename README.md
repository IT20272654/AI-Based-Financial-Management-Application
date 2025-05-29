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


<img width="1423" alt="Screenshot 2025-05-30 at 01 20 01" src="https://github.com/user-attachments/assets/7377f203-496a-4ac5-b628-5d9e6b86611a" />
<img width="1411" alt="Screenshot 2025-05-30 at 01 20 22" src="https://github.com/user-attachments/assets/993c62c3-c91e-4b15-be9b-3604d8cb3d24" />
<img width="1414" alt="Screenshot 2025-05-30 at 01 20 34" src="https://github.com/user-attachments/assets/4912f9a7-71e9-4ea9-8f86-9d8c66a4c9e8" />
<img width="1418" alt="Screenshot 2025-05-30 at 01 20 42" src="https://github.com/user-attachments/assets/efbc51b9-7993-4b79-ab18-8fc036a0ba00" />
<img width="1416" alt="Screenshot 2025-05-30 at 01 21 07" src="https://github.com/user-attachments/assets/4d791948-41d0-4f05-8c42-036cdedc45fd" />
<img width="1420" alt="Screenshot 2025-05-30 at 01 21 27" src="https://github.com/user-attachments/assets/ab675599-4f66-4b5e-ab22-d287a226e3de" />
<img width="1420" alt="Screenshot 2025-05-30 at 01 21 36" src="https://github.com/user-attachments/assets/8fd6895d-d027-405b-9b3c-0bb2231a6857" />















