

# something 1, 2 , 3 , example, cr7and messi r files were also used 






library(dplyr)
library(stats)
library(psych)
library(ggfortify)
library(tidyverse)
library(ggplot2)
library(Hmisc)
library(PerformanceAnalytics)
library(stringr)  

library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)

#install and call library
library(rpart)
library(caret)
library(dplyr)
library(ggplot2)
library(reshape2)
library(mltools)
library(e1071)
library(class)


pf<- read.csv("C:/Users/Aniket/Desktop/Disseration/final_Yt_dataset.csv",na.strings=c(""))
tail(pf,3)
str(pf)

# Dataset pre-processing 

# Convert text to lowercase
text_data_lower <- str_to_lower(pf$title)

#creating category of games 1 for rolepaly 2 for others and 3 for pubg

pf$category <- ifelse(grepl("tlrp|roleplay|rp|gta", pf$title), 1, 2)
pf$category <- ifelse(grepl("bgmi|pubg", pf$title), 3, pf$category)

pf



summary(pf)




#finding likes 
which.max(pf$likeCount)
which.min(pf$likeCount)

#finding views
rownames(pf)[which.max(pf$viewCount)]
rownames(pf)[which.min(pf$viewCount)]

# changing duration in seconds
x <- pf$duration

x2 <- sapply(x, function(i){
  t <- 0
  if(grepl("S", i)) t <- t + as.numeric(gsub("^(.*)PT|^(.*)M|^(.*)H|S$", "", i))
  if(grepl("M", i)) t <- t + as.numeric(gsub("^(.*)PT|^(.*)H|M(.*)$", "",i)) * 60
  if(grepl("H", i)) t <- t + as.numeric(gsub("^(.*)PT|H(.*)$", "",i)) * 3600
  t
})

pf$durationSecs = x2
pf$X <- NULL


# Cleaning the data one last time
# Check for missing values in a dataframe 
missing_values <- is.na(pf)

# Count missing values in each column 
missing_counts <- colSums(is.na(pf))

# Check for complete cases 
complete_cases <- complete.cases(pf)

# Impute missing values
pf$favouriteCount[is.na(pf$favouriteCount)] <- 1
pf$tags[is.na(pf$tags)] <- 'No Tags!!!'
pf$description[is.na(pf$description)] <- 'Sorry, forgot to write Description!!!'
pf$likeCount[is.na(pf$likeCount)] <- 0

# Removing outliers
# Assuming 'data' is your dataset and 'likes' is the numerical variable
likes <- pf$likeCount

# Calculate the interquartile range (IQR)
Q1 <- quantile(likes, 0.25, na.rm = TRUE)
Q3 <- quantile(likes, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1

# Define the lower and upper thresholds for outliers
lower_threshold <- Q1 - 1.5 * IQR
upper_threshold <- Q3 + 1.5 * IQR

# Identify outliers
outliers <- likes < lower_threshold | likes > upper_threshold

# Create a subset without outliers 
clean_data <- pf[!outliers, ]

#Taking useful data for analysis 
selected <- clean_data[ ,c('viewCount','likeCount','commentCount','category')]
not_selected <-clean_data[,c('video_id','channelTitle','title','description','tags','publishedAt','favouriteCount','duration','caption','definition','Time','durationSecs')]

correlation_matrix <- cor(selected)

# Visualize correlation matrix as a heatmap
melted_correlation <- melt(correlation_matrix)
ggplot(data = melted_correlation, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "green") +
  labs(title = "Correlation Heatmap")
#note for me in wordfile(comments) as well
#"Likes" and "Views" have a strong positive correlation, indicated by the green color (correlation coefficient close to +1). This implies that as the number of likes increases, the number of views also tends to increase. These two variables are positively associated and may provide similar information regarding engagement with the videos."Comments" shows a moderate positive correlation (correlation coefficient of approximately 0.5) with "Likes," "Views," and "Category." This suggests that there is a moderate relationship between the number of comments and these variables. It implies that videos with more likes, views, and belonging to a specific category tend to receive a higher number of comments.The length of the video has a moderate correlation (correlation coefficient of approximately 0.5) with "Likes" and "Views." This implies that the length of the video may have some influence on the number of likes and views it receives. Longer videos might attract more engagement, but this correlation is not as strong as the correlation between likes and views.


# Analyzing with 'Category' and 'Published At' columns

# Convert 'Published Date' column to date format if needed
clean_data$publishedAt <- as.Date(clean_data$publishedAt)

# Calculate the frequency of categories by published date
category_freq <- table(clean_data$category, clean_data$publishedAt)

# Convert 'Published Date' column to date format 
clean_data$publishedAt <- as.Date(clean_data$publishedAt)

# Create a new column for the month and year
clean_data$Month_Year <- format(clean_data$publishedAt, "%Y-%m")

# Calculate the frequency of categories per month
category_freq <- table(clean_data$category, clean_data$Month_Year)

# Print the frequency of categories per month
for (category in unique(clean_data$category)) {
  category_freq_month <- category_freq[category, ]
  print(paste("Category", category, "is published", sum(category_freq_month), "times within the following months:"))
  print(category_freq_month)
  print("")
}

#ploting  frequency of categories per month
barplot(category_freq, beside = TRUE, legend.text = c('Roleplay','Other games','Pubg'),
        main = "Frequency of Categories by Month",
        xlab = "Month", ylab = "Frequency", col = rainbow(nrow(category_freq)))

# Feature selection category as target variable and like, view, comment count as input variable 

# Perform label encoding
encoded_category <- as.numeric(as.factor(clean_data$category))
# Calculate the average likes, views, and comments for each category
category_avg <- aggregate(cbind(likeCount, viewCount, commentCount) ~ category, clean_data, FUN = mean)

# Merge the average values back with the original dataset
clean_data <- merge(clean_data, category_avg, by = "category", suffixes = c("", "_avg"))

# Use the average values as encoded features
encoded_category <- clean_data[, c("likeCount", "viewCount", "commentCount")]

# Splitting data into test and train data 
# Set the random seed for reproducibility
set.seed(42)

# Specify the proportion of data for testing (e.g., 20% for testing)
test_ratio <- 0.2

# Perform train-test split
train_indices <- createDataPartition(selected$category, p = test_ratio, list = FALSE)
train_data <- selected[train_indices, ]
test_data <- selected[-train_indices, ]

# Building model for prediction


#Decision Tree
# Combining the input features and target variable into a single dataframe
final <- cbind(train_data[, c("likeCount", "viewCount", "commentCount")], Category = train_data$category)

# Training the Decision Tree model
tree_model <- rpart(Category ~ ., data = final, method = "class")

# Print the Decision Tree model summary
print(tree_model)

# Evaluation metrics
# Make predictions on the test data
test_data_features <- test_data[, c("likeCount", "viewCount", "commentCount")]
predictions <- predict(tree_model, newdata = test_data_features, type = "class")

predictions <- factor(predictions, levels = unique(test_data$category))
test_data$category <- factor(test_data$category, levels = unique(test_data$category))

# Building a confusion matrix
confusion_mat <- confusionMatrix(predictions, test_data$category)

# Extracting evaluation metrics
accuracy <- confusion_mat$overall["Accuracy"]
precision <- confusion_mat$byClass[,"Precision"]
recall <- confusion_mat$byClass[,"Recall"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the evaluation metrics results
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")

# Create a vector of evaluation metric values
metrics <- c(Accuracy = accuracy, Precision = precision, Recall = recall, `F1-score` = f1_score)

# Ploting the evaluation metrics
barplot(metrics, ylim = c(0, 1), main = "Decision Tree Model Evaluation Metrics", ylab = "Value", col = "blue")


#Knn
# Combining the input features and target variable into a single dataframe
final <- cbind(train_data[, c("likeCount", "viewCount", "commentCount")], Category = train_data$category)
# Convert the target variable to a factor
train_data$category<- factor(train_data$category)

# Training the Random Forest model
knn_model <- knn(train = final[, -4], test = test_data[, c("likeCount", "viewCount", "commentCount")], cl = final[, 4], k = 5)

knn_model <- factor(knn_model, levels = unique(test_data$category))
test_data$category <- factor(test_data$category, levels = unique(test_data$category))

# Building a confusion matrix
confusion_mat <- confusionMatrix(knn_model, test_data$category)

# Extracting evaluation metrics
accuracy <- confusion_mat$overall["Accuracy"]
precision <- confusion_mat$byClass[,"Precision"]
recall <- confusion_mat$byClass[,"Recall"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Printing the evaluation metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")

# Creating a vector of evaluation metric values
metrics <- c(Accuracy = accuracy, Precision = precision, Recall = recall, `F1-score` = f1_score)

# Ploting the evaluation metrics
barplot(metrics, ylim = c(0, 1), main = "Knn Model Evaluation Metrics", ylab = "Value", col = "blue")

# Combine the input features into a single dataframe
final<- train_data[, c("likeCount", "viewCount", "commentCount")]

# K-means 
# Perform k-means clustering
k <- 3  # Specify the number of clusters
kmeans_model <- kmeans(final, centers = k, nstart = 25)

# Get the cluster assignments for the data points
cluster_assignments <- kmeans_model$cluster

# Printing the cluster assignments
print(cluster_assignments)

# Combining the input features and cluster assignments into a single dataframe
clustered_data <- cbind(final, Cluster = as.factor(cluster_assignments))

# Calculate the centroid coordinates for each cluster
centroids <- kmeans_model$centers

# Creating a dataframe for the centroids
centroid_data <- data.frame(Center = factor(1:k), x = centroids[, 1], y = centroids[, 2])

# Ploting the clusters and centroids
ggplot() +
  geom_point(data = clustered_data, aes(x = likeCount, y = viewCount, color = Cluster)) +
  geom_point(data = centroid_data, aes(x = x, y = y, color = Center), size = 7) +
  geom_polygon(data = centroid_data, aes(x = x, y = y, r = 5), color = "black", fill = NA) +
  labs(title = "K-Means Clustering with Circles", x = "Likes", y = "Views") +
  scale_color_discrete(name = "Cluster")


#SVM
# Combine the input features and target variable into a single dataframe
final <- cbind(train_data[, c("likeCount", "viewCount", "commentCount")], Category = train_data$category)

# Training the SVM model
svm_model <- svm(Category ~ ., data = final, kernel = "linear")

# Make predictions on the test data
test_data_features <- test_data[, c("likeCount", "viewCount", "commentCount")]
predictions <- predict(svm_model, newdata = test_data_features)

predictions <- factor(predictions, levels = unique(test_data$category))
test_data$category <- factor(test_data$category, levels = unique(test_data$category))

# Building a confusion matrix
confusion_mat <- confusionMatrix(predictions, test_data$category)

# Extracting evaluation metrics
accuracy <- confusion_mat$overall["Accuracy"]
precision <- confusion_mat$byClass[,"Precision"]
recall <- confusion_mat$byClass[,"Recall"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Printing the evaluation metrics results
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")


# Create a vector of evaluation metric values
metrics <- c(Accuracy = accuracy, Precision = precision, Recall = recall, `F1-score` = f1_score)

# Ploting the evaluation metrics
barplot(metrics, ylim = c(0, 1), main = "SVM Model Evaluation Metrics", ylab = "Value", col = "blue")


#write clean data
write.csv(pf, 'ytclean_FinalRsdata.csv')



