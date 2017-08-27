# Question type prediction

## Goal
Identify Question Type: Given a question, the aim is to identify the category it belongs to. The four categories are : Who, What, When, Affirmation(yes/no).
Label any sentence that does not fall in any of the above four as "Unknown" type.

Example:
``1. What is your name? Type: What`` <br />
``2. When is the show happening? Type: When`` <br />
``3. Is there a cab available for airport? Type: Affirmation`` <br />
There are ambiguous cases to handle as well like: <br />
``What time does the train leave(this looks like a what question but is actually a When type)``

## Testing
For testing, you can also look for datasets on the net. Sample (though the categories are different here): http://cogcomp.cs.illinois.edu/Data/QA/QC/train_1000.label

## Approaches <br />
1) Approach1: Bruteforce approach <br />
  This is not a machine learning based approach, but a simple rule based, which I understood by analyzing the data.  <br />
  But it gives very high accuracy of 98% <br />
   <br />
2) Approach2: Attention Layer Based approach  <br />
	Attention Layer identifies which words are important in the message. Implemented a Attention Layer on the Subject msg.  <br />
	Trained the on the data with multiple epochs, but got an accuracy of 11% <br />
	With this approach, I could not achieve good results. <br />
   <br />
3) Approach3: RandomForest ( and Decision Tree) based approach <br />
  Converted the Subject msg to sequences format using text_to_sequences module and prepared X and y data. <br />
  Executed RandomForestClassifier on the data and got accuracy of average 78%. <br />
   <br />
  
## Instructions to run the code:
```
1) pip install -r requirements.txt
2) Change the file paths in the code 
3) Execute the python file
```
