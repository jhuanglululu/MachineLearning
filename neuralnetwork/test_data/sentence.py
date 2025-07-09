def square_question(number):
    return f'null what is the square of {number} null'.split(' ')

def square_root_question(number):
    return f'null what is the square root of {number} null'.split(' ')

def sum_question(number1, number2):
    return f'null what is the square of {number1} and {number2} null'.split(' ')

def square_answer(number):
    return f'null the square of {number} is {number ** 2} null'.split(' ')

def square_root_answer(number):
    return f'null the square root of {number} is {int(number ** 0.5)} null'.split(' ')

def sum_answer(number1, number2):
    return f'null the square of {number1} and {number2} is {number1 + number2} null'.split(' ')

def square_question_answer(number):
    return square_question(number) + square_answer(number)

def square_root_question_answer(number):
    return square_root_question(number) + square_root_answer(number)

def sum_question_answer(number1, number2):
    return sum_question(number1, number2) + sum_answer(number1, number2)