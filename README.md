# Unsupervised Machine Learning for Mental Health Analysis

## Project Overview
This project aims to analyze mental health data using unsupervised machine learning techniques. The goal is to identify distinct clusters of survey respondents based on their experiences, diagnoses, roles in the technology industry, and more. The insights derived from this analysis can help HR departments implement targeted mental health support programs.

## Dataset
The dataset consists of survey responses from technology employees regarding their mental health experiences. It includes categorical, numerical, and binary variables, such as mental health diagnoses, work environment factors, and demographic information. The dataset can be found and downloaded on https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016.

## Objectives
- **Data Cleaning & Preprocessing:** Handle missing values, mapped textual inputs, and extracted topics.
- **Clustering:** Use unsupervised learning techniques to identify distinct clusters in the dataset.
- **Feature Analysis:** Identify the strongest parameters defining each cluster.
- **Visualization:** Provide clear visual representations of distributions, correlations, and cluster characteristics.

## Key Steps in Analysis
1. **Data Cleaning:**
   - Topic extraction of open questions (diagnoses).
   - Mapped categorical variables to meaningful groups.
   - Handled missing values and unknown responses.
   
2. **Exploratory Data Analysis (EDA):**
   - Visualized distributions of roles, diagnoses, and topics.
   - Identified the most common concerns among respondents.
   
3. **Clustering Analysis:**
   - Applied clustering algorithms to group similar respondents.
   - Identified the most defining features of each cluster.
   
4. **Feature Importance Analysis:**
   - Determined the two strongest parameters defining each cluster.
   - Analyzed the role of employment history (Column 24) in clustering.
   
5. **Visualizations:**
   - Bar charts, pie charts, and heatmaps for insights into the dataset.
   - Age distribution, regional distributions, diagnose distribution, role distribution and correlation matrices.

## Results
The findings are summarized in the [Results.md](Results.md) file, which includes:
- Analysis of the strongest parameters for each cluster.
- Insights into age, regional distributions, and mental health concerns.
- Recommendations for HR interventions based on clustering insights.

## How to Run the Code
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Phips92/UnsupervisedML_MentalHealth_IU.git
   cd UnsupervisedML_MentalHealth_IU
   ```
   
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run EDA and Clustering Analysis:**
   ```bash
   python analysis.py
   ```
   
4. **Generate Visualizations:**
   ```bash
   python visualization.py
   ```


## Contributions
Contributions are welcome! Feel free to fork this repository, create issues, or submit pull requests.

   - Fork the repository.
   - Create a new branch (git checkout -b feature/YourFeature).
   - Commit your changes (git commit -m "Add your feature").
   - Push to the branch (git push origin feature/YourFeature).
   - Open a pull request.


## License
This project is licensed under the GNU General Public License v3.0


## Author
For questions or feedback, feel free to contact me at [philipp92.mcguire@gmail.com].



