class NumberList:
    
    def __init__(self):
        self.numbers = []

    def insert_number(self, num):
        self.numbers.append(num)

    def search_number(self, target):
        if target in self.numbers:
            return self.numbers.index(target) +1
        else:
            return -1
        

def main():
    num_list = NumberList()
    ## Read a positive integer number
    try:
        N= int (input ("Enter a positive integer number of elements: "))
        if N <= 0:
            print("Please enter a positive integer.")
            return
    except ValueError:
        print("Please enter a valid positive integer.")
        return
    
    for i in range(N):
        num = int(input(f"Enter number {i+1}: "))
        num_list.insert_number(num)

    X = int(input("Enter the number want to search for: "))

    result = num_list.search_number(X)
    print(result)

if __name__ == "__main__":
    main()