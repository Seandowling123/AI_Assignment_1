# Example list of values
my_list = ['key1', 'value1', 'key2', 'value2', 'key3', 'value3']

# Initialize an empty dictionary
result_dict = {}

# Iterate over the list and construct the dictionary
for i in range(0, len(my_list)):
    # Check if there's at least one more element after the current index
    if i + 1 < len(my_list):
        # Add key-value pair to the dictionary
        result_dict[my_list[i]] = my_list[i + 1]

print(result_dict)
