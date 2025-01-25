# zadané hodnoty
first_value = 23
median = 69
var_range = 200
average = 98
n = 99
# doplnění hodnot do pole
data = [0] * n
data[0] = first_value
data[n//2] = median
data[n-1] = first_value + var_range

target_sum = average * n
current_sum = sum(data)
remaining_sum = target_sum - current_sum

# projde pole a doplní na další pozici stejnou hodnotu jako předchozí
for i in range(n):
    if data[i] == 0:
        data[i] = data[i - 1]
    i += 1

# projde pole zezadu a doplní zbývající hodnoty aby vycházel průměr
if current_sum != target_sum:
    for i in range(n):
        i = n-1
        while i > 0:
            i -= 1
            if i != n//2 and i != n and i != 0:
                while target_sum - sum(data) != 0:
                    if data[i] < data[i+1]:
                        data[i] += 1
                    else:
                        break
# výpis výsledného pole
print(data)
# výpis zbytku (pokud je jiiný než 0, úloha nemá řešení)
print(target_sum - sum(data))