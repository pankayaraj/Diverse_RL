import heapq

# define the priority queue list
pq = []

# function to push an element to the priority queue
def push_array_with_priority(array, priority):
    heapq.heappush(pq, (priority, array))

# function to pop the element with the highest priority
def pop_array_with_priority():
    _, array = heapq.heappop(pq)
    return array

# example usage
push_array_with_priority([1, 2, 3], -2)
push_array_with_priority([4, 5, 6], -1)
push_array_with_priority([7, 8, 9], -22)

print(heapq.heapreplace(pq, ( -12, [1, 2, 3])))
print(pq)
print(pop_array_with_priority()) # output: [4, 5, 6]
print(pop_array_with_priority()) # output: [1, 2, 3]
print(pop_array_with_priority()) 