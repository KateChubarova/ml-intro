class AstroBody:
    description = 'Natural entity in the observable universe.'


def square_args(func):
    def inner(a, b):
        return func(a ** 2, b ** 2)

    return inner


@square_args
def multiply(a, b):
    return a * b


print(multiply(3, 9))

lst = [1, 2, 3, 4]

lst_iterator = iter(lst)

print(next(lst_iterator))

from sklearn import tree

clf = tree.DecisionTreeClassifier()
