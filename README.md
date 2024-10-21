### YouTube Data Analysis Project
## Overview
This project leverages the YouTube Data API to extract statistics and details from various YouTube channels, specifically focusing on the following channels:
- AntismartDevil
-	Ani
-	Dusskit Gaming
-	GG GAMING

The extracted data includes channel statistics, video IDs, and detailed information about each video, such as view counts, like counts, and publication dates. The resulting dataset can be used for further analysis, visualization, or machine learning applications.

## Table of Contents
-	Overview
-	Installation
-	Usage
-	Data Extraction
-	Data Visualization
-	Output Data
-	Libraries Used
-	License

## Installation
To set up this project, you need to have Python installed along with the required libraries. You can clone this repository and install the necessary dependencies as follows:

## Requirements
-	Python 3.6 or later
-	Google API Client Library
-	Pandas
-	Seaborn

## Usage
-	Get your API Key: Sign up for the Google Developer Console, create a new project, and enable the YouTube Data API v3 to get your API key. Replace the placeholder api_key with your actual API key in the code.
-	Run the Script: Execute the script to gather data from the specified YouTube channels.
-	Output: The collected data will be saved as Yt_data.csv in the specified directory.

## Data Extraction
The project extracts data using the following steps:
-	Channel Statistics: The get_channel_stats function retrieves statistics for the specified channels, including:
    -	Channel Name
    -	Subscriber Count
     -	Total Views
     -	Total Video Count
     - Video IDs: The get_video_ids function collects video IDs from the channels' playlists
- Video Details: The get_video_details function fetches detailed statistics for each video, including:
   -	Title
   -	Description
   -	View Count
   -	Like Count
   -	Comment Count
   -	Published Date
- Data Visualization: Data visualization is performed using the Seaborn library to create bar plots for:
   -	Subscriber counts
   -	View counts
   -	Total video counts
- Output Data
The final output is a CSV file named Yt_data.csv, containing:
  -	Video ID
  -	Channel Title
  -	Video Title
  -	Description
  -	Published Date
  -	View Count
  -	Like Count
  -	Comment Count
This file can be used for further analysis or as a dataset for machine learning models.

## Libraries Used
This project uses the following libraries:
- googleapiclient: For interacting with the YouTube Data API.
-	pandas: For data manipulation and analysis.
-	seaborn: For data visualization.

## Additional Notes:
-	Make sure to handle your API key securely and not expose it publicly.
-	You may want to set a quota for API requests to avoid hitting limits.
-	This project can be extended by adding more features, such as real-time updates, data analysis, or visualizations based on the collected data.
Feel free to modify any sections to better suit your project! If you need any more assistance or clarification, just let me know!

