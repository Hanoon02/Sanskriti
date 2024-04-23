# Sanskriti - Conversing Cultures

## Setup

1. **Clone the repository into your local system:**

   ```bash
   git clone https://github.com/Hanoon02/Sanskriti.git
   cd Sanskriti
   ```

2. **Set up the virtual environment**

   ```bash
   virtualenv sans
   python3 -m venv sans
   # For Windows
   sans\Scripts\activate
   # For Mac
   source sans/bin/activate
   ```

3. **Install the packages and backend data**

   ```bash
   pip install -r requirements.txt
   python3 setup.py
   ```

4. Run Sanskriti application

   ```bash
   cd Application
   python3 app.py
   ```

The application will be running on `http://127.0.0.1:5000`.

Verified Python Versions
1. Python 3.10.6  