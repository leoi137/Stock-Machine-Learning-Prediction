# README

This is a basic collection of modules that allow for the creation of multiple machine learning models in order to predict stock future stock prices. This uses a random forest algorithm by default but it can work with any machine learning algorithm. Note that this works with any sort of asset data it uses yahoo stock data for simplicity.

# HOW IT WORKS

1. Gathers stock data from the yahoo API
2. Creates various features using the StockFeatureBuilder module
3. Uses the FeatureSelection module to gather the K best features
4. Runs multiple random forest models and chooses the best one
5. Shows the important features sorted by their importance
6. Runs a K-Fold cross validation
7. Tests and outputs results and saves model and required data