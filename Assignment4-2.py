N = int(input("Enter a positive integer N: "))

# Read N numbers one by one
numbers =[]
for i in range(N):
    num = int(input(f"Enter number {i+1}: "))
    numbers.append(num)

# Search number
X = int(input("Enter a number X to search for: "))

# Search X in the list
if X in numbers:
    index = numbers.index(X) + 1
    print(index)
else:
    print(-1)
