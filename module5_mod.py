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