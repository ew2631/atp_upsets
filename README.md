# upset_the_atp
### Predicting the outcome of ATP matches

## Abstract

My goal is to predict the outcome of ATP Tennis matches. I’ll be using the dataset of all ATP matches from 2000 to 2018 with roughly 45,000 matches and 23 variables. By using match and player statistics, the ability to predict match outcomes would be incredibly powerful for determining future match draws, sports betting opportunities, and tournament tactics for players

## Design

This project originates from a  Kaggle dataset and notebook "Beat the bookmaker with machine learning". The data is provided by Edouard Thom. Each row in the dataset represents one professional ATP match, details about the location and size of the tournament, and details on the match-the winner, the loser, and how many sets each player won or lost. The target variable is “upset”, coded as 1 if a lower ranked player beats a higher ranked player (i.e. Player w/ Rank #48 beats Player w/ Rank #4). This was chosen for 2 reasons: First, the intuitive choice for the target variable is win or loss, but since that is a unique string (a player’s name), I would have to create two rows for each match, one row for the winner and one row for the loser. Second, for specific sports betting use cases, it is most important to predict for upsets because those occurrences provide the best upside. 

## Data

The dataset contains roughly 45,000 matches and 23 variables. There are 35% upsets and 65% not upset matches. Some notable variables include: 
