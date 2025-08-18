# Movie Recommendation

This project is a web application built with Gradio that recommends movies based on content similarity (genre, plot, cast, director, etc.). Similarity calculation is done using TF-IDF and cosine similarity.

## 🛠️ Tools and Programming Language Used

* **Language**: Python 3.9
* **Main Lybraries**: Gradio, Pandas, WordCloud
* **Containerization**: Docker

## How to Run the Project (Recommended: Docker)

The simplest and most guaranteed way to run this application is using Docker, as it takes care of all the dependencies and configurations.

### Prerequisites

* [Git](https://git-scm.com/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## ⚙️ Data Config

Before execute the project, it is necessary to download the dataset.

1. Do the download of "TMDB 5000 Movie Dataset" from Kaggle:
    * **Link:** [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

2. After download, extract the zip file.

3. Create a folder in THE ROOT called: `data/raw/`.

4. Copy the files `tmdb_5000_credits.csv` and `tmdc_5000_moveis.csv` inside the folder `data/raw/`.

### Steps

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/VictorHFerreira016/movie-recommendation.git](https://github.com/VictorHFerreira016/movie-recommendation.git)
    ```

2.  **Navigate to the project folder:**
    ```bash
    cd RECOMMENDATION PROJECT
    ```

3.  **Build the Docker image:**
    *This command will download the dependencies and build the application image. It may take a few minutes the first time.*
    ```bash
    docker build -t app-recomendacao .
    ```

4.  **Run the Docker container:**
    *This command will start the application and map the required port.*
    ```bash
    docker run -p 7860:7860 app-recomendacao
    ```

5.  **Access the application**
    Open your browser and access the URL:
    [**http://localhost:7860**](http://localhost:7860)

---

## How to Run the Project (Locally with Virtual Environment)

If you prefer not to use Docker, you can run the application locally.

### Prerequisites

* Python 3.9 or superior
* Git

### Steps

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/VictorHFerreira016/movie-recommendation.git](https://github.com/VictorHFerreira016/movie-recommendation.git)
    ```

2.  **Navigate to the project folder:**
    ```bash
    cd RECOMMENDATION PROJECT
    ```

3.  **Create and activate the enviroment:**
    *On Windows:*
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    *On macOS/Linux:*
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Execute the Gradio**
    ```bash
    python interface.py
    ```

6.  **Acess the application**
    Click on the link:
    ```bash
    Gradio will provide a local URL in your terminal. Access it in your browser:
    `http://127.0.0.1:7860`
    ```

## 📁 Estrutura do Projeto

```
.
├── Dockerfile              # Defines the Docker application environment
├── requirements.txt        # List of Python dependencies
├── streamlit_app.py         # Main code for the Gradio web application
├── scripts/                 # Modules with preprocessing and modeling logic
│   ├── preprocessing.py
│   └── utils.py
└── README.md                # This file :)
```

## 👤 Author

**[Victor Hugo]**

* LinkedIn: [Victor Hugo](https://www.linkedin.com/in/victor-hugo-0a7821300/)
* GitHub: [@VictorHFerreira016](https://github.com/VictorHFerreira016)