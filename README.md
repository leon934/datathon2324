# Data Cleanser

This is a simple data cleansing tool that parses information from a hospital's CSV file and converts it into useful information for machine learning algorithms aimed to predict the probability of a patient dying.

## Unnecessary Columns
First, the tool removes unnecessary columns that should not effect the outcome of the prediction. These columns include (but are not limited to):
- Cost associated with administrator time (dollars)
- Size of dose of medicine, since this data was the same for every patient
- Education and income of patient
- Day of announcement of important updates

## Reformatting and Parsing
Then, the tool reforms some columns, such as the `sex` column which has mixed data (male, Male, M, female, and 1).

The tool also judges values that are out of the standard expected range, such as a heart rate in the thousands, and removes invalid data.

It also parses categorical data into numeric formats. This data comes from the R^2 test performed on the specific data against the death of a patient. 

## Statistical Analysis
Finally, the tool also performs statistical analysis and sees if specific columns have a correlation with the death of a patient. This step is done manually and the specific columns must be analyzed by a human to see if they are relevant to the prediction. Columns with bad correlation can be removed in the first step mentioned above.


# Usage
This tool is written in Python and requires the following libraries:
- pandas
- numpy
- matplotlib
- scikit-learn
