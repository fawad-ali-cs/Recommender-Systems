import numpy as np

def basic_MF(R, training_values, latent_features=10, steps=100, alpha=0.01, reg=0.15):
    P = np.random.rand(R.shape[0], latent_features)
    Q = np.random.rand(R.shape[1], latent_features)
    
    for _ in range(steps):
        for u,i in training_values:
            e = R[u, i] - np.dot(P[u], Q[i])
            P[u] = P[u] + alpha*(e*Q[i] - reg*P[u])
            Q[i] = Q[i] + alpha*(e*P[u] - reg*Q[i])
        
    return P, Q


def bias_MF(R, training_values, latent_features=10, steps=100, alpha=0.01, reg=0.15):
    P = np.random.rand(R.shape[0], latent_features)
    Q = np.random.rand(R.shape[1], latent_features)
    
    bias_user = np.random.rand(R.shape[0])
    bias_movie = np.random.rand(R.shape[1])
    
    total_rating = 0
    for u,i in training_values:
        total_rating += R[u, i]
    
    avg_rating = total_rating/len(training_values)
    
    for _ in range(steps):
        for u,i in training_values:
            e = R[u, i] - np.dot(P[u], Q[i]) - avg_rating - bias_user[u] - bias_movie[i]
            
            P[u] = P[u] + alpha*(e*Q[i] - reg*P[u])
            Q[i] = Q[i] + alpha*(e*P[u] - reg*Q[i])
            bias_user[u] = bias_user[u] + alpha*(e - reg*bias_user[u])
            bias_movie[i] = bias_movie[i] + alpha*(e - reg*bias_movie[i])
        
    return P, Q, avg_rating, bias_user, bias_movie
