from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes", split="train")
print(dataset[0])
print("="*10)
print(dataset.num_rows)
print("="*10)
print(dataset)
print("="*10)

# Define the old and new column names
old_column_name = 'text'  # Replace with the column you want to rename
new_column_name = 'text_new'  # Replace with the new column name

# Rename the column and keep only the renamed column
dataset = dataset.map(lambda x: {new_column_name: x[old_column_name]}, 
                      remove_columns=dataset.column_names)

# Print the modified dataset
print(dataset[0])
print("="*10)
print(dataset.num_rows)
print("="*10)
print(dataset)
print("="*10)

print("done")