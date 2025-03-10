"""Don't add unneeded functionality to your code"""
def calculate_final_price(prices):
    """
    Print the total discounted price. The discount is fixed at 10% 
    
    INPUT
    ------
    prices  :   list
                list of price values
    
    OUTPUT
    ------
    out_str :   str         
                formatted text 
    """

    total = sum(prices)

    return f"${0.9*total}"


prices = [4, 7, 3, 6, None, 8, 12, 19]
#calculate_final_price(prices)


"""Rewrite code to follow Curly's Law"""
def clean_data(data):
    #Removes None values from the data.
    return [x for x in data if x is not None]

def average_data(data):
    # Returns the average over the data list
    return sum(data)/len(data)

def format_to_str(avg):
    return f"Average: {avg:.2f}"

def process_data(data):
    # cleans the data and prints out the average.
    
    cleaned_data = clean_data(data)
    average = average_data(cleaned_data)
    text = format_to_str(average)
    return text

#print(process_data(prices))

"""Avoid repeated code! Replace with a function and/or a for loop"""
def format_name(name):
    # prints "Hello, name!" for each name in the list
    return f"Hello, {name}!"

names_lst = ["Alice", "Bob", "Charlie"]
for name in names_lst:
    print(format_name(name))

"""Rewrite the code following POLA, removing unnecessary functionality and throw errors where expected"""
from math import pi

def rectangle_area(length, width):
    if length < 0 or width < 0:
        raise ValueError('Negative values given!')
    
    return length*width

def circle_area(radius):
    if radius < 0:
        raise ValueError('Negative value given!')
    
    return pi*radius**2

def triangle_area(base, height):
    if base < 0 or height < 0:
        raise ValueError('Negative values given!')
   
    return 0.5*base*height
