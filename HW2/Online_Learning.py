def cal(str):
    number0 = 0
    number1 = 0
    for i in range(len(str)):
        for j in range(len(str[i])):
            if (str[i][j] == "0"):
                number0 += 1
            if (str[i][j] == "1"):
                number1 += 1
    # print(number1)
    # # print(number1)
    return number0, number1


def factorial(num):
    if num < 0:
        print("negative num does not exist")

    elif num == 0:
        return 1

    else:
        fact = 1
        while (num > 1):
            fact *= num
            num -= 1
        return fact


def Online_learning(a, b):
    for i in range(len(case)):
        print("case ", i+1, ": ", case[i][0])
        number0, number1 = cal(case[i])
        total_number = number0+number1
        # print(total_number)
        c = factorial(total_number)/(factorial(number0)*factorial(number1))
        likelihood = c*((number1/total_number)**number1) * \
            ((number0/total_number)**number0)
        print("Likelihood:", likelihood)
        print("Beta prior:     a =", a, "b =", b)
        a += number1
        b += number0
        print("Beta posterior: a =", a, "b =", b, "\n")


path = "testfile.txt"
case = []
with open(path, 'r') as f:
    for line in f.read().splitlines():
        s = line.split('\n')
        case.append(s)

a = input("a : ")
a = int(a)
b = input("b : ")
b = int(b)
Online_learning(a, b)
