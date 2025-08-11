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
    
    # Put the positive integer in the list
    for i in range(N):
        try:
            num = int(input(f"Enter positive integer number {i+1}: "))
            if num <= 0:
                print("Please enter a positive integer.")
                return
            num_list.insert_number(num)
        except ValueError:
            print("Please enter a valid positive integer.")
            return

    X = int(input("Enter the number want to search for: "))

    #Search the result
    result = num_list.search_number(X)
    print(result)

if __name__ == "__main__":
    main()
