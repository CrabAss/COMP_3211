import random

def partition(array, left, right):
    pivot = left
    for i in range(left+1, right+1):
        if array[i] <= array[left]:
            pivot += 1
            array[i], array[pivot] = array[pivot], array[i]
    array[pivot], array[left] = array[left], array[pivot]
    return pivot

def _qsort(arr,left,right):
	if (left >= right):
		return
	p=partition(arr,left,right)
	_qsort(arr,left,p-1)
	_qsort(arr,p+1,right)
	if(random.randint(0,100)<5):
		foo(arr) # bug

def foo(arr):
	arr[0] , arr[-1] = arr[-1] ,arr[0]


def qsort(arr):
	length = len(arr)
	if length<=1:
		return arr
	_qsort(arr,0,len(arr)-1)
	return arr

	

if __name__ == "__main__":
	print(qsort([5,4,3,2,1,0,-1,6,7,8,9,10]))