# Homework 1

This assignment is designed to guide you in building a comprehensive time series analysis framework from the ground up. By personally implementing core components—including data loading, preprocessing, feature transformation, model construction, and evaluation—you will gain a deep understanding of the workings and implementation details of classical time series models (e.g., ARIMA, DLinear).

The homework spans six key modules:

1. **Dataset Handling & Visualization**: Learn to efficiently load and explore time series data.
2. **Data Transformations**: Implement crucial preprocessing techniques like standardization and normalization.
3. **Evaluation Metrics**: Programmatically compute standard metrics such as MAE and MAPE.
4. **Forecasting Models**: Progress from implementing simple baselines to advanced deep learning models.
5. **Decomposition Techniques**: Explore how decomposition methods can enhance model performance.
6. **Advanced Model (Bonus)**: Challenge yourself by implementing cutting-edge research models like PatchTST and iTransformer.

We hope this homework will not only solidify your theoretical knowledge but also sharpen your skills in engineering practice and experimental analysis. Please read each part's requirements carefully and enjoy the process of building your very own time series analysis framework!



## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --channels 7 --target OT --model MeanForecast
```



## Part 1. Dataset (20 pts)

path: , `dataset/dataset.py`, `dataset/data_visualizer.py`

All datasets can be found [here](https://box.nju.edu.cn/d/b33a9f73813048b8b00f/).

**Objective:** In this part, you will implement a custom dataset class `CustomDataset` , to handle various time series datasets. Your custom dataset class will inherit from a base class `DatasetBase` , and provide functionality to read and preprocess the data. This assignment aims to enhance your understanding of dataset handling in machine learning and time series analysis.

**Instructions:**

**1. CustomDataset Class:**

- Implement the method `read_data` in the class:
  - It should handle different dataset formats (CSV files) by providing a general mechanism for reading and preprocessing the data.
  - Note that the format and structure of the dataset may vary, so ensure your implementation is flexible.
- Implement the method `split_data` in the class:
  - Split the data into training, validation, and test sets based on the ratios provided in the object. You can use the `args.ratio_train`, `args.ratio_val` and `args.ratio_test` attributes from the base class `DatasetBase`.

**2. `data_visualize` Function:**

- Implement the method `data_visualize` in `data_visualizer.py`.
- This function should take two parameters:
  - `dataset`: Dataset to visualize. Dataset.data might have shape (n_samples, timesteps, channels) or (n_samples, timesteps), depending on whether the dataset has multiple channels.
  - `t`: An integer representing the number of continuous time points to visualize.
- Ensure that your implementation is flexible enough to handle both single-channel and multi-channel datasets.

**3. Testing:**

- Test your implementation by creating an instance of the class `CustomDataset` and loading a custom dataset using the function `get_dataset`.
- Verify that the data is read and split correctly by accessing the relevant attributes of the dataset object (e.g.`dataset.train_data` ).
- After loading the dataset, call the function `data_visualize` on the dataset object with a specific value of `t` to visualize the data.

**In your report, plot one of the provided datasets and analyze the dataset's properties briefly according to the images. You can design the way you plot your dataset freely to make the images more beautiful or easier to analyze.**



## Part 2. Transform (10 pts)

path: `utils/transforms.py`

**Objective:** In this part, you will implement various data transformation techniques for preprocessing time series data. These transformations will help prepare the data for machine learning models. You will create custom transformation classes that inherit from the base class `Transform` and implement the necessary methods.

**Instructions:**

**1. Transform Base Class:**

You are provided with a base class named `Transform`, which defines the structure for data transformations. This class has two abstract methods: `transform` and `inverse_transform`. You will create custom transformation classes that inherit from this base class and implement these methods.

**2. Custom Transformation Classes:**

You need to implement the following custom transformation classes, each inheriting from the base class `Transform`. Make sure that all transformation class can handle multivariate data.

**a. Normalization**

**b. Standardization**

**c. MeanNormalization**

**d. BoxCox** (input might less than 0)

**In your report, write down the mathematical formula for each transformation.**



## Part 3 Metrics (10 pts)

path: `utils/metrics.py`

**Objective:** In this programming assignment, you will implement several metrics commonly used to evaluate the performance of predictive models. These metrics will help you assess the accuracy and quality of predictions made by models. Your task is to implement the Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (SMAPE), and Mean Absolute Scaled Error (MASE) metrics.

**In your report, write down the mathematical formula for each metric.**







## Part 4 Models (40 pts)

path: `models/baselines.py`,  `models/ARIMA.py` , `models/DLinear.py`

**Objective:** In this part, you will design and implement four forecasting models: Linear Regression, Exponential Smoothing, ARIMA and DLinear. These models will take historical time series data of length `seq_len` as input and generate predictions of length `pred_len`.

**Instructions:**

**1. MLForecastModel Base Class:**

You are provided with a base class named `MLForecastModel`, which defines the structure for forecasting models. 

**2.  Models:**

**a. LR:** Linear Regression (`models/baselines.py`)

**b. ES:** Exponential Smoothing (`models/baselines.py`)

**c. ARIMA** ( `models/ARIMA.py`)

**d. DLinear:** Implement the DLinear model as described in the provided [paper](https://arxiv.org/pdf/2205.13504.pdf). You can define the dataloader yourself or modify if necessary. ( `models/DLinear.py`)

**3. Testing:**

- Test all models and on the dataset you plot in Part 1.
- Load the selected datasets, split them into training and testing sets, and apply each forecasting model to make predictions on the testing data.
- Calculate and record relevant metrics for each model and dataset pair, including Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (SMAPE), and Mean Absolute Scaled Error (MASE).
- Present the evaluation metrics in a table format, showcasing the performance of each model on each dataset.

**In your report, test all algorithm with different transformations on the dataset you plot in Part 1 and fill the table below.**

| Dataset | Model | Transform | MSE  | MAE  | MAPE | SMAPE | MASE |
| ------- | ----- | --------- | ---- | ---- | ---- | ----- | ---- |
| dataset | AR    | None      |      |      |      |       |      |
|         |       | Normalize |      |      |      |       |      |
|         |       | Box-Cox   |      |      |      |       |      |
|         |       | ...       |      |      |      |       |      |
|         | ES    | None      |      |      |      |       |      |
|         |       | Normalize |      |      |      |       |      |
|         |       | Box-Cox   |      |      |      |       |      |
|         |       | ...       |      |      |      |       |      |



## Part 5 Decomposition (20 pts)

path: `utils/decomposition.py`

**Objective:** Decomposition is a common technique in time series analysis that can transform the structure of data. With proper application, it may reduce the difficulty of model processing, thereby leading to performance improvements. Implement time series decomposition methods to separate the trend and seasonal components from the original time series data and integrate these methods into the DLinear and other forecasting models you have.

**Instructions:**

**1. Moving Average Decomposition**

Implement the moving_average function that calculates the trend and seasonal components using a moving average with a specified seasonal period.

**2. Differential Decomposition**

Implement the differential_decomposition function that separates the trend and seasonal components by differencing the time series data. Determine how to calculate the differences and reconstruct the trend and seasonal components from these differences.

**3. STL**

Implement STL decomposition methods to separate the trend and seasonal components from the original time series data.

**In your report, documenting whether the application of various decomposition methods to different models leads to performance improvements.**



## Part 6 More Advanced Model* (bonus 20pts)

path: `models/Your_Model.py`

**objective:** In this part, you will implement more advanced time series analysis methods, such as [PatchTST](https://arxiv.org/abs/2211.14730), [iTransformer](https://arxiv.org/abs/2310.06625), or other models that demonstrate superior performance to DLinear. These models feature more ingenious designs or more complex structures, enabling them to handle time series problems more effectively compared to baseline methods.

**Instructions:**

- Test the advanced model on 2-3 of the real-world time series datasets, such as ETT and Custom datasets, to evaluate its performance.
- Use evaluation metrics (e.g., MAE, MAPE, SMAPE, MASE) to measure the model's accuracy and compare it with the basic version of the model.
- Compare the computational time and memory required for various settings.

**In your report, include the detailed implement of your algorithm and comparison of time and accuracy in 2-3 of the provided datasets.**



## Submission

**1. Modified Code:**

- Provide the modified code for all components of the task.
- Include a file in Markdown format that covers the entire task. This file should contain `README.md`: 
  - how to install any necessary dependencies for the entire task.
  - how to run the code and scripts to reproduce the reported results.
  - datasets used for testing in all parts of the task.

**2. PDF Report:**

- Create a detailed PDF report that encompasses the entire task. The report should include sections for each component of the task.

**3. Submission Format:**

- Submit the entire task, including all code and scripts, along with the file `README.md` and the PDF report, in a compressed archive (.zip).

**4. Submission Deadline:** 2025-12-02 23:55
