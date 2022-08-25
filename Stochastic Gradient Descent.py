import numpy as np
import math
import random
trials = 30
N = 400
d = 5
M = 2
Rho = math.sqrt(2)

def Projection(vector):
  d = np.linalg.norm(vector)
  if d>1 :
    return vector/d
  return vector

def GenerateData(sigma, T):
  X = []
  Y = []
  for i in range(T):
    y = random.choice([-1, 1])
    x = np.random.normal(y/4, sigma, d-1)
    Y.append(y)
    X.append(x)
  return X, Y

def ComputeWs(sigma, T):
  X, Y = GenerateData(sigma, T)
  # initialization: w1 = 0
  rows, cols = (T+1, d)
  W = [[0]*cols]*rows
  Ws = np.zeros(d)
  Learning_Rate = M / (Rho * math.sqrt(T))
  for t in range(1, T):
    # <x,1>
    Xt = np.append(Projection(X[t]), [1], axis = 0)

    # y<w,~x>
    z = Y[t] * np.dot(np.array(W[t]), Xt)

    # logistic loss function
    # loss = math.log(1 + math.exp(-z))

    # Compute Gt
    Gt = (-1 * math.exp(-z) / (1 + math.exp(-z))) * Y[t] * Xt

    # Update wt+1
    W[t+1] = W[t] - Learning_Rate * Gt
    W[t+1] = Projection(W[t+1])

  for t in range(1, T+1):
    Ws = np.add(Ws, W[t]) 
  return Ws/T

def GetLossAndError(Ws, x, y_test):
  Error = 0
  Loss = 0
  for i in range(N):
    x_test = np.append(Projection(x[i]), [1], axis = 0)
    z = y_test[i] * np.dot(np.array(Ws), x_test)
    loss = math.log(1 + math.exp(-z))
    Loss+=loss

    true_y = y_test[i]
    if np.dot(x_test, Ws) >0:
      pred_y = 1
    else: 
      pred_y = -1

    if pred_y!=true_y:
      Error+=1
  
  Loss = Loss/N
  Error = Error/N

  return Loss, Error

def SGD_Learner():
  print("{0:^21}|{1:^31}|{2:^20}".format("", "Logistic loss", "Classification error"))
  print("{0:^4}|{1:^4}|{2:^3}|{3:^7}|{4:^5}|{5:^7}|{6:^5}|{7:^11}|{8:^10}|{9:^10}"
  .format("σ","n", "N","#trials","Mean","Std Dev", "Min", "Excess Risk", "Mean", "Std Dev"))
  print("--------------------------------------------------------------------------")

  for sigma in [0.1, 0.35]:
    x_test, y_test = GenerateData(sigma, N)#generate test set
    for n in [50, 100, 500, 1000]:
      Total_Loss = []
      Total_Error = []
      for t in range(30):
        Ws = ComputeWs(sigma, n)
        Loss, Error = GetLossAndError(Ws, x_test, y_test)
        Total_Loss.append(Loss)
        Total_Error.append(Error)
      L_mean = sum(Total_Loss)/len(Total_Loss)
      L_std = np.std(Total_Loss)
      L_min = min(Total_Loss)
      L_excess_risk = L_mean-L_min

      C_mean = sum(Total_Error)/len(Total_Error)
      C_std = np.std(Total_Error) 

      print("{0:^4}|{1:^4}|400|{2:^7}|{3:^5.3f}|{4:^7.3f}|{5:^5.3f}|{6:^11.3f}|{7:^10.3f}|{8:^10.3f}"
      .format(sigma, n, 30, L_mean, L_std, L_min, L_excess_risk, C_mean, C_std))
    print("--------------------------------------------------------------------------")

def main():
  SGD_Learner()

if __name__ == "__main__":
    main()
