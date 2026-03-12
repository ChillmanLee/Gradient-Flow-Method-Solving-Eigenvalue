from enum import Flag
from tabnanny import check
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import gammaln


print("=====This is designed for solving the maximum eigenvalue======")
print("\n")
print("=================Designed by Chillman Lee=====================")

Debug=0
def debug(Debug):

    if Debug==1:
        print("A worng input in get_matrix")

    if Debug==2:
        print("A worng input in initialize_x")
        
def get_matrix(scale=1000,upper_bound=1,lower_bound=0,distribution="uniform"):

    if distribution=="normal":

        A=np.random.normal(lower_bound,upper_bound,size=(scale,scale))

    elif distribution=="uniform":

        A=np.random.uniform(lower_bound,upper_bound,size=(scale,scale))

    else:
        Debug=1
        debug(Debug)

    A=(A+A.T)/2

    return A

def initialize_x(A,scale=10000,distribution="uniform"):

    n=len(A)

    if distribution=="uniform":

        X=np.random.uniform(0,1,size=(n,scale))

    elif distribution=="normal":

        X=np.random.normal(0,1,size=(n,scale))

    else:
        Debug=2
        debug(Debug)
        return

    X=X/np.linalg.norm(X,axis=0)

    E=np.diag(np.dot(X.T,np.dot(A,X)))

    max_index=np.argmax(E)

    x=X[:,max_index].T

    return x

def Grad(A,x):
    grad = 2 * (np.dot(A, x) - np.dot(x.T, np.dot(A, x)) * x)
    return grad

def Renew(grad,x,lr):
    y=x+lr*grad
    return y

def train(A,x,Max_Step=5000,lr=0.001,memory_strength=0.9,check_value=1e-15,epsilon=1e-32):
    steps=0
    grad_memory=0
    result_residue=1e32
    result_eigenvalue=-1e32
    for steps in range(1,Max_Step):


        grad_now=Grad(A,x)
        grad_memory=grad_now+memory_strength*grad_memory


        if np.linalg.norm(grad_memory) < epsilon:

             break

        x=Renew(grad_memory,x,lr)
        x=x/np.linalg.norm(x)
    

        eigenvalue = np.dot(x.T, np.dot(A, x)) 
        residue=np.linalg.norm(np.dot(A,x)-eigenvalue*x)
        
        result_eigenvalue=max(result_eigenvalue,eigenvalue)
        result_residue=min(result_residue,residue)

        if(residue < check_value):
            break
    return result_eigenvalue,result_residue,x

def main():
   A=get_matrix()

   # print("Matrix\n")
   # print(A)

   x=initialize_x(A)

   # print("Initial vector\n")
   # print(x)

   e,r,x=train(A,x)

   print("result\n")
   print(e)
   print("residue\n")
   print(r)
   # print("vector\n")
   # print(x)
main()
