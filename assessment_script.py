from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col,trim, to_timestamp
from pyspark.sql.types import StringType, TimestampType
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

# Creating a Spark session
spark = SparkSession.builder.appName("InterviewDataImport").getOrCreate()

# Define the path to the raw_data_interview folder
path = "./raw_data_interview/"
student_df = spark.read.option("header", "true").csv(path + "student_data.csv", inferSchema=True)
course_df = spark.read.option("header", "true").csv(path + "course_data.csv", inferSchema=True)
login_df = spark.read.option("header", "true").csv(path + "student_logins.csv", inferSchema=True)
content_df = spark.read.option("header", "true").csv(path + "content_access.csv", inferSchema=True)
assignment_df = spark.read.option("header", "true").csv(path + "assignment_data.csv", inferSchema=True)
submission_df = spark.read.option("header", "true").csv(path + "assignment_submissions.csv", inferSchema=True)

#transformation/cleaning
def clean_dataframe(df: DataFrame) -> DataFrame:
    try:
        for column, dtype in df.dtypes:
            # 1. Handle null values based on data type
            if dtype in ['int', 'bigint', 'double', 'float', 'long', 'decimal']:
                df = df.fillna({column: 0})
            elif dtype in ['string']:
                df = df.fillna({column: ''})
            elif dtype in ['timestamp', 'date']:
                df = df.fillna({column: '1970-01-01 00:00:00'})

        # 2. Normalize timestamp columns
        for field in df.schema.fields:
            if isinstance(field.dataType, TimestampType):
                df = df.withColumn(field.name, to_timestamp(col(field.name), 'yyyy-MM-dd HH:mm:ss'))

        # 3. Generic Cleaning
        # - Trim strings
        for field in df.schema.fields:
            if isinstance(field.dataType, StringType):
                df = df.withColumn(field.name, trim(col(field.name)))

        return df
    except Exception as e:
        print(f"An error occurred while cleaning the DataFrame: {e}")
        raise

transformed_student_df = clean_dataframe(student_df).toPandas()
transformed_course_df = clean_dataframe(course_df).toPandas()
transformed_login_df = clean_dataframe(login_df).toPandas()
transformed_content_df = clean_dataframe(content_df).toPandas()
transformed_assignment_df = clean_dataframe(assignment_df).toPandas()
transformed_submission_df = clean_dataframe(submission_df).toPandas()

#------------------------------------------------------------------------------------------------------->
##1. How students engage with course content
# Plotting distribution of time spent on different content types
# Calculate the average time spent on each content type
average_time_spent = transformed_content_df.groupby('content_type')['time_spent_minutes'].mean().reset_index()
average_time_spent.columns = ['content_type', 'average_time_spent']

# Plotting the distribution of time spent on different content types with values on top of bars
plt.figure(figsize=(12, 6))
sns.barplot(x='average_time_spent', y='content_type', data=average_time_spent, orient='h')
plt.title('Average Time Spent on Each Content Type')
plt.xlabel('Average Time Spent (Minutes)')
plt.ylabel('Content Type')

# Add values on top of the bars
for index, row in average_time_spent.iterrows():
    plt.text(row['average_time_spent'], index, f'{row["average_time_spent"]:.2f}', va='center', ha='right')

plt.savefig('Average time spent by students on each content type.png')
plt.show()

##visualization for which department spent the highest time on which type of content
# Merge transformed_course_df and transformed_content_df based on course_id
merged_df = transformed_course_df.merge(transformed_content_df, on='course_id')

# Group by content_type and department, then sum the time_spent_minutes
grouped_df = merged_df.groupby(['content_type', 'department'])['time_spent_minutes'].sum().reset_index()

# Pivot the dataframe to have departments as columns
pivot_df = grouped_df.pivot(index='content_type', columns='department', values='time_spent_minutes')

# Plot the pivot table
pivot_df.plot(kind='bar', stacked=True)
plt.xlabel('Content Type')
plt.ylabel('Total Time Spent (minutes)')
plt.title('Time Spent by Department on Each Content Type')
plt.legend(title='Department')
plt.savefig('Time Spent by Department on Each Content Type.png')
plt.show()

# #------------------------------------------------------------------------------------------------------->
# #2. Patterns in assignment completion and grades
# Distribution of Grades Across Assignments
# Filter the transformed_submission_df to include only the columns needed
submission_filtered = transformed_submission_df[['assignment_id', 'status', 'points']]
assignment_filtered = transformed_assignment_df[['assignment_id','max_points']]

# Calculate the grade for each assignment
submission_filtered['grade'] = submission_filtered['points'] / assignment_filtered['max_points'] * 100

# Group by assignment_id and calculate the mean grade
mean_grades = submission_filtered.groupby('assignment_id')['grade'].mean().reset_index()

# Create a histogram to visualize the distribution of grades
plt.figure(figsize=(10, 6))
plt.hist(mean_grades['grade'], bins=20, edgecolor='black')
plt.title('Distribution of Grades Across Assignments')
plt.xlabel('Grade (%)')
plt.ylabel('Number of Assignments')
plt.grid(True)
plt.savefig('Distribution of Grades Across Assignments.png')
plt.show()

# # Count the occurrences of each status
status_counts = submission_filtered['status'].value_counts()

# Create a bar chart to visualize the counts
plt.figure(figsize=(10, 6))

# Plot the bar chart
status_counts.plot(kind='bar')

# Add the counts on top of each bar
for i, v in enumerate(status_counts):
 plt.text(i, v + 5, str(v), ha='center')

plt.title('Count of Assignments by Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('Count the occurrences of each status.png')
plt.show()

# #------------------------------------------------------------------------------------------------------->
# #3. Correlation between engagement metrics and academic performance
# Total logins and avg session duration
login_engagement = transformed_login_df.groupby('student_id').agg(
    total_logins=('login_id', 'count'),
    avg_session_duration=('session_duration_minutes', 'mean')
).reset_index()

# Content engagement
content_engagement = transformed_content_df.groupby('student_id').agg(
    total_content_accessed=('access_id', 'count'),
    total_time_spent=('time_spent_minutes', 'sum')
).reset_index()

# Merge submission with assignment to get max_points
merged_submissions = transformed_submission_df.merge(
    transformed_assignment_df[['assignment_id', 'max_points', 'due_date']],
    on='assignment_id'
)

# Convert timestamp and due_date to datetime
merged_submissions['timestamp'] = pd.to_datetime(merged_submissions['timestamp'])
merged_submissions['due_date'] = pd.to_datetime(merged_submissions['due_date'])

# Create performance metrics
merged_submissions['on_time'] = merged_submissions['timestamp'] <= merged_submissions['due_date']

performance_metrics = merged_submissions.groupby('student_id').agg(
    avg_score=('points', 'mean'),
    submission_count=('submission_id', 'count'),
    on_time_rate=('on_time', 'mean')
).reset_index()

dfs = [login_engagement, content_engagement, performance_metrics]
student_insights_df = reduce(lambda left, right: pd.merge(left, right, on='student_id', how='outer'), dfs)
student_insights_df = student_insights_df.fillna(0)

# Correlation Heatmap (Engagement vs. Performance)
plt.figure(figsize=(10, 6))
sns.heatmap(student_insights_df.drop('student_id', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Engagement Metrics and Academic Performance")
plt.tight_layout()
plt.savefig('Correlation Heatmap (Engagement vs. Performance).png')
plt.show()

# Scatter Plot: Total Time Spent vs. Average Score

plt.figure(figsize=(8, 5))
sns.scatterplot(data=student_insights_df, x='total_time_spent', y='avg_score')
plt.title("Total Time Spent vs. Average Score")
plt.xlabel("Total Time Spent on Content (minutes)")
plt.ylabel("Average Assignment Score")
plt.tight_layout()
plt.savefig('Scatter Plot: Total Time Spent vs. Average Score.png')
plt.show()

# Scatter Plot: Total Logins vs. On-Time Submission Rate

plt.figure(figsize=(8, 5))
sns.scatterplot(data=student_insights_df, x='total_logins', y='on_time_rate')
plt.title("Total Logins vs. On-Time Submission Rate")
plt.xlabel("Total Logins")
plt.ylabel("On-Time Submission Rate")
plt.tight_layout()
plt.savefig('Scatter Plot: Total Logins vs. On-Time Submission Rate.png')
plt.show()
# #------------------------------------------------------------------------------------------------------->
# # 4. Potential early warning indicators for students at risk
# Identify students who have not submitted assignments
at_risk_students = merged_df[merged_df['status'] == 'pending'].groupby('student_id').size().reset_index(name='count_incomplete_submissions')
at_risk_students = at_risk_students[at_risk_students['count_incomplete_submissions'] > 0]

# Plot the distribution of incomplete submissions
plt.figure(figsize=(10, 6))
sns.histplot(at_risk_students['count_incomplete_submissions'], bins=20, kde=True)
plt.title('Distribution of Incomplete Submissions')
plt.xlabel('Number of Incomplete Submissions')
plt.ylabel('Frequency')
plt.savefig('Distribution of Incomplete Submissions.png')
plt.show()

# Plot the average time spent on content per student
average_time_spent = merged_df.groupby('student_id')['time_spent_minutes'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.boxplot(x='student_id', y='time_spent_minutes', data=average_time_spent)
plt.title('Average Time Spent on Content Per Student')
plt.xlabel('Student ID')
plt.ylabel('Time Spent (Minutes)')
plt.savefig('Average Time Spent on Content Per Student.png')
plt.show()
