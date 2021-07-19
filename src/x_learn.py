# pi = 3.14159
# print(pi)
from datetime import datetime, timedelta

currrent_now = datetime.now()
# timedelta is used to define a period of time
# one_Day = timedelta(days=1)
# yesterday = currrent_now - one_Day
# birthday = input("When is your birthday (dd/mm/yy) ")
# birthday_date = datetime.strptime(birthday, '%d/%m/%Y')
# print(f"Birthday: {birthday_date}")
x = 2
y = 0
try:
    print(x/y)
finally:
    print("faild")

