import torch
'''
https://rogerspy.github.io/2021/09/12/einsum-mhsa/
'''
# A = [[1,2,3],[4,5,6]]
# B = [[7,8],[9,10], [11,12]]
# torch.einsum()

a = torch.arange(6).reshape(2, 3)
print(a)
b = torch.einsum('ij -> ji', a)
print(b)

splitor = '#####' * 5


print(splitor)
a = torch.arange(6).reshape(2, 3)
print(torch.einsum('ij ->', a))

print(splitor)

print(torch.einsum('ij -> j', a))
print(torch.einsum('ij -> i', a))

print(splitor)

a = torch.arange(6).reshape(2, 3)
b = torch.arange(3)
print(torch.einsum('ik, k -> i', a, b))

print(splitor)
a = torch.arange(6).reshape(2, 3)
b = torch.arange(15).reshape(3, 5)
print(torch.einsum('ik,kj -> ij', a, b))

print(splitor)
a = torch.arange(3)
b = torch.arange(3, 6)
print(torch.einsum('i,i ->', a, b))

print(splitor)
a = torch.arange(6).reshape(2, 3)
b = torch.arange(6, 12).reshape(2, 3)
print(torch.einsum('ij, ij -> ij', a, b))

print(splitor)
a = torch.arange(3)
b = torch.arange(3, 7)
print(torch.einsum('i, j -> ij', a, b))

print(splitor)
a = torch.randn(3, 2, 5)
b = torch.randn(3, 5, 3)

print(torch.einsum('ijk, ikl -> ijl', a, b)) # TODO check logic
print(splitor)

a = torch.randn(2,3,5,7)
b = torch.randn(11,13,3,17,5)
print(torch.einsum('pqrs, tuqvr -> pstuv', a, b).shape) # TODO check logic