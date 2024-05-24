from scipy.stats import friedmanchisquare
# Intersec
# avranks = [5.46, 8.00, 5.69, 6.85, 4.00, 2.77, 1.92, 1.31]

# Cosine
# avranks = [5.76, 7.92, 5.15, 6.85, 4.23, 2.77, 1.85, 1.31]

# KL
# avranks = [5.69, 8.00, 5.46, 6.85, 3.92, 2.77, 2.00, 1.23]

# Clark
# avranks = [5.38, 8.00, 5.62, 7.00, 4.00, 2.62, 2.00, 1.31]

# Canber
# avranks = [5.23, 8.00, 5.77, 7.00, 4.00, 2.77, 1.92, 1.31]

# cheb
# avranks = [6.00, 8.00, 5.00, 6.85, 4.08, 2.69, 2.08, 1.23]


# euc
avranks = [8.0,4.7,4.1,5.9,5.9,3.3,2.1,2.0]

# sren
# avranks = [7.9,4.3,3.7,6.0,6.1,3.6,2.4,1.8]

# squa
# avranks = [7.9,4.7,3.7,5.8,6.2,3.8,2.5,1.4]

# cheby
# avranks = [7.8,4.5,3.9,5.7,6.3,4.1,2.2,2.0]

# intersec
# avranks = [7.9,4.3,3.7,5.9,6.2,3.6,2.5,1.7]

# fide
# avranks = [7.9,4.7,3.1,6.0,6.1,3.6,2.6,1.2]

# cosine
# avranks = [7.8,5.1,4.0,6.1,5.7,3.2,1.8,1.9]
k = 8
n = 10
ri = 0
for i in avranks:
    ri += i ** 2
# res = (12 * n) / (k * (k + 1)) * (ri - (k * (k + 1) ** 2) / 4)
# final_res = (n - 1) * res / (n * (k - 1) - res)
print(ri)
result_Tx2 = (12 * n) * (ri - ((k * (k + 1) ** 2) / 4)) / (k * (k + 1))
print(result_Tx2)
result_Tf = (n - 1) * result_Tx2 / (n * (k - 1) - result_Tx2)

print(result_Tf)
