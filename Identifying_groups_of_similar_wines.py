
'''ASSIGNMENT: IDENTIFYING GROUPS OF SIMILAR WINES''' 

# Importing libraries
import numpy as np
import pandas as pd


# Initializing class
class Matrix:
    # Initializing the Matrix object with a 2D array
    def __init__(self, array_2d):
        
        self.array_2d = np.array(array_2d)  # Convert the input data to a numpy array
        self.rows, self.cols = self.array_2d.shape  # Get the shape of the array

    @staticmethod
    # Loading data from a CSV file
    def load_from_csv(file_name):
        
        data_frame = pd.read_csv(file_name)  # Read the CSV file using pandas
        return Matrix(data_frame.values)  # Convert it to a Matrix object


    # Standardise the matrix data using the formula:
    # D'_ij = (D_ij - mean(D_j)) / (max(D_j) - min(D_j))
    def standardise(self):
        
        standardised_data = np.zeros_like(self.array_2d, dtype=float)  # Initializing an array for standardized values
        for j in range(self.cols):  # Iterating through each column
            column = self.array_2d[:, j]  # Extracting column data
            mean_col = np.mean(column)  # Calculate mean of the column
            max_col = np.max(column)  # Getting the maximum value in the column
            min_col = np.min(column)  # Getting the minimum value in the column
            if max_col - min_col != 0:  # Avoiding division by zero
                standardised_data[:, j] = (column - mean_col) / (max_col - min_col)  # Standardise
        return Matrix(standardised_data)  # Returning a new Matrix with the standardized data


    # Calculating the Euclidean distance between row i of this matrix and each row in another matrix.
    # Returns the distances as a column vector.
    def get_distance(self, other_matrix, row_i):
        
        row_i_data = self.array_2d[row_i]  # Getting the row_i data from the matrix
        distances = np.sqrt(np.sum((other_matrix.array_2d - row_i_data) ** 2, axis=1))  # Calculateing Euclidean distance
        return distances.reshape(-1, 1)  # Returning the distances as a column vector


    # Calculating the Weighted Euclidean distance between row i of this matrix and each row in another matrix.
    # The weights array is applied to each dimension
    def get_weighted_distance(self, other_matrix, weights, row_i):
        
        row_i_data = self.array_2d[row_i]  # Getting the row_i data from the matrix
        weighted_distances = np.sqrt(np.sum(weights * (other_matrix.array_2d - row_i_data) ** 2, axis=1))  # Calculateing Weighted distance
        return weighted_distances.reshape(-1, 1)  # Returning the distances as a column vector
    
    
    # Counting the frequency of unique elements in the matrix and return a dictionary
    def get_count_frequency(self):
        
        unique, counts = np.unique(self.array_2d, return_counts=True)  # Getting unique values and their counts
        return dict(zip(unique, counts))  # Returning the results as a dictionary


# Functions outside the class 

# Initializing a weight vector of length m where the sum of the weights is 1
def get_initial_weights(m):
   
    weights = np.random.rand(m)  # Generating random weights
    return weights / np.sum(weights)  # Normalizing the weights to sum to 1


# Computing centroids for K clusters. Each centroid is the mean of all rows assigned to a cluster.
# Returning the centroids as a Matrix object.
def get_centroids(matrix, S, K):
    
    centroids = np.zeros((K, matrix.cols))  # Initializing a matrix for centroids
    for k in range(K):
        cluster_rows = matrix.array_2d[S == k]  # Getting all rows assigned to cluster k
        if len(cluster_rows) > 0:
            centroids[k] = np.mean(cluster_rows, axis=0)  # Computing the mean of the rows
    return Matrix(centroids)


#Calculating the within-cluster separation for each dimension.
def get_separation_within(matrix, centroids, S, K):
    
    m = matrix.cols  # Number of dimensions
    separation_within = np.zeros((1, m))  # Initializing the separation matrix
    for j in range(m):
        for i in range(matrix.rows):
            u_ik = 1 if S[i] == j else 0  # Indicator for cluster assignment
            centroid_k = centroids.array_2d[S[i], j]  # Getting the centroid of the cluster
            separation_within[0, j] += u_ik * np.linalg.norm(matrix.array_2d[i, j] - centroid_k) ** 2  # Calculating within-cluster separation
    return separation_within


# Calculate the between-cluster separation for each dimension
def get_separation_between(matrix, centroids, S, K):
   
    m = matrix.cols  # Number of dimensions
    separation_between = np.zeros((1, m))  # Initializing the separation matrix
    for j in range(m):
        for k in range(K):
            Nk = np.sum(S == k)  # Counting the number of rows in cluster k
            if Nk > 0:
                separation_between[0, j] += Nk * np.linalg.norm(centroids.array_2d[k, j] - matrix.array_2d[:, j].mean()) ** 2  # Calculating between-cluster separation
    return separation_between


# Assigning each row in the matrix to a random cluster, creating the group assignment array S.
def get_groups(matrix, K):
    
    S = np.random.randint(0, K, size=matrix.rows)  # Random initialization of cluster assignments
    return S


# Updating the weights based on the separation within and between clusters.
def get_new_weights(centroids, separation_within, separation_between, old_weights, S, K):
    
    new_weights = np.zeros_like(old_weights)  # Initializing the new weights array
    for j in range(len(old_weights)):
        sum_term = np.sum(separation_between[0, j] / separation_within[0, j] for j in range(len(old_weights)))  # Summation term in the weight update formula
        new_weights[j] = 0.5 * (old_weights[j] + (separation_between[0, j] / (separation_within[0, j] * sum_term)))  # Updating each weight
    return new_weights


# Test run function

# Testing the algorithm for printing the frequency of unique elements from provided data.
def run_test():
    
    m = Matrix.load_from_csv('Data (2).csv')  # Loading the matrix from a CSV file
    for k in range(2, 11):  # Looping over different numbers of clusters
        for i in range(20):  # Performing 20 iterations for each cluster size
            S = get_groups(m, k)  # Randomly assigning groups
            print(f"Clusters: {k}, Frequency: {m.get_count_frequency()}")  # Printing the result


# Example usage :
file_path = r"C:\Users\rahul\Jupyter Documents\Identifying_groups_of_similar_wines_Python_Task_Anubavam\Data (2).csv"  # Specify the path to your CSV file
data_matrix = Matrix.load_from_csv(file_path)  # Load data into the Matrix object

# Run the test
run_test()  # Execute the test function



# Conclusion:
    
# The code provides a framework for implementing and testing clustering algorithms, with a focus on:
# 1.Standardizing data.
# 2.Calculating distances (Euclidean and weighted).
# 3.Performing clustering operations such as random group assignment, calculating centroids, and updating weights based on separation measures.




