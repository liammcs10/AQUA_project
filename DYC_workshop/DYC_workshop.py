

class Fruit:
    # object representing a single fruit
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity
    
    def display(self):
        print(f'There are {self.quantity}  {self.name}s')

    def get_totalValue(self):
        print(f'{self.name} :  {str(self.price*self.quantity)}')

class Shop:
    # object representing a whole selection of fruits
    def __init__(self, fruits):
        # fruits is a list of fruit objects
        self.stock = fruits

    def display_stock(self):
        for fruit in self.stock:
            fruit.display()
    
    def get_total_stock_value(self):
        total = 0
        for fruit in self.stock:
            total += fruit.price*fruit.quantity
        print(f'The total value of the shop is {total}')

fruits = [
    Fruit("apple", 1.0, 10),
    Fruit("banana", 0.5, 20),
    Fruit("orange", 0.75, 15)
]

shop = Shop(fruits)

shop.display_stock()
shop.get_total_stock_value()
        