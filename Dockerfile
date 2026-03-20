# 1. Select the base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files (like your Covid Data.csv) into the container
COPY . .

# 6. Open a port so we can access Jupyter Notebook from our web browser
EXPOSE 8888

# 7. Set the default command to start Jupyter Notebook when the container runs
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]