Solutions for BMW Tasks

Task 1:
Your objective is to develop a chatbot system powered by large language model (LLM)
inference on your local machine taking advantage of GPU acceleration to
optimize performance and including these machine details in your submission
documentation. If you don't have access to a machine with GPU, CPU inference is also sufficient. 
The interface of the chatbot should be designed for ease of use by human
users. Establish a local git repository on your machine for version control, and employ
professional best practices by committing updates regularly as you progress through the
task. Select a suitable LLM for the chatbot and consider integrating established
packages from trusted sources such as public GitHub repositories or PyPI. In your
documentation, justify your selection of the model and packages. Ensure the repository
remains local and not uploaded to any public platforms like GitHub.

----------------------------------------------------

Solution = Here I developed General Chatbot that answers the questions specific to uploaded pdf documents. Task1_app.py is the file.

-----------------------------------------------------

Task 2:
This task is aiming to provide a binary classification of the column “Type”. 
The dataset is provided in two different tables ("table_1.csv" and "table_2.csv") 
with unique identifier of column “ID”. Tip: This column (ID) can be used to match the 
two tables. 

-----------------------------------------------------

Solution: Used various Machine Learning models for Binary Classification. Task2_BinaryClass_Project.ipynb is the Solution file.

-----------------------------------------------------

Task 3:
The file “Parts.csv” contains descriptions of some fictitious parts. Your goal is to 
find 5 alternative parts to each provided fictitious part in the dataset based on their 
similarity. First provide descriptive analysis of the data and highlight 2-3 findings 
and difficulties of the data that we provided and describe how you would handle this. 
Continue to implement a solution that is finding the similar fictitious parts based on 
the column “DESCRIPTION”. Please give details of your solution and why you choose it. 
Once you finished your implementation of your solution, please think about how you 
would integrate your code into the chatbot from task 1. 

-----------------------------------------------------

Solution: Here I developed Two kinds of Scripts. 1. Streamlit Chatbot application Task3_main.py (Here because of Resource constraint, Model inference was a struggle) 2. Task3_Mistral_Project.ipynb  This file is a Colab Jupyter notebook where i used GPU for Mistral model inference. Here model responds to the user questions.  

-----------------------------------------------------

Task 4:
Write a function that takes as input two timestamps of the form 2017/05/13 12:00 and 
calculates their differences in hours. Please only return the full hour difference and 
round the results. E.g. 2022/02/15 00:05 and 2022/02/15 01:00 would return 1 hour.

Task 5:
Expand the above function to only count the time difference between 09:00 – 17:00 and 
only on weekdays.

-----------------------------------------------------

Solution: Task4_Task5.ipynb file haev solution for Task4 ans Task5.

-----------------------------------------------------
