import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def process(excel_file):
    def process_sheet_J(excel_file):
        # Load Excel file into a pandas ExcelFile object
        xls = pd.ExcelFile(excel_file)

        pd.set_option('display.max_rows', None)

        # Get the sheet names
        sheet_names = xls.sheet_names

        # Create an empty dictionary to store DataFrames
        dfs = {}

        for sheet_name in sheet_names:
            # Read data from the current sheet into a pandas DataFrame
            df = pd.read_excel(excel_file, sheet_name=sheet_name, na_values=['NaN', 'NA', '', None])

            # Store the DataFrame in the dictionary with the sheet name as key
            dfs[sheet_name] = df

        # Accessing DataFrame for sheet "J"
        df_sheet_main = dfs["main"]

        # Selecting the desired columns
        selected_columns = ['WORKSHOP', 'EQUIPMENT_GROUP', 'EQUIPMENT_SUB_GROUP', 'EQUIPMENT', 'PRODUCT',
                            'WAFERS_COUNT',
                            'ALARM_DURING_PROCESS', 'REAL_THROUGHPUT_IN_WAFERS_PER_HOUR']

        # Creating a new DataFrame with the selected columns
        df_sheet_main_reduce = df_sheet_main[selected_columns].copy()

        # Remove rows with missing values (NaNs)
        df_sheet_main_reduce.dropna(inplace=True)

        # Save the DataFrame to an Excel file
        df_sheet_main_reduce.to_excel('CPTO_reduce.xlsx', index=False)

    # Call the function with the Excel file path as argument
    process_sheet_J('CPTO_2.xlsx')

    # Load Excel file into a pandas ExcelFile object
    excel_file = 'CPTO_reduce.xlsx'
    pd.set_option('display.max_rows', None)

    df = pd.read_excel(excel_file)

    def df_display(df_name):
        print(df_name.shape)
        print(df_name.head())
        print(df_name.tail(1))

    df['ALARM_DURING_PROCESS'] = df['ALARM_DURING_PROCESS'].map({'Y': 2, 'N': 1})
    # Extracting numbers after underscore for the specified columns
    columns_to_process = ['WORKSHOP', 'EQUIPMENT_GROUP', 'EQUIPMENT_SUB_GROUP', 'EQUIPMENT', 'PRODUCT']

    for column in columns_to_process:
        df[column] = df[column].str.split('_').str[-1]  # Get the last part after splitting
        df[column] = pd.to_numeric(df[column],
                                   errors='coerce')  # Convert to numeric, handle errors by setting them as NaN

    df_display(df)
    print(df.dtypes)

    df.to_excel('CPTO_clean.xlsx', index=False)


excel_file = 'CPTO_FOR_PY.xlsx'
pd.set_option('display.max_rows', None)

df = pd.read_excel(excel_file)
#sample_df = pd.read_excel(excel_file)
#df = sample_df.sample(frac=0.2)


def df_display(df_name):
    print(df_name.shape)
    print(df_name.head())
    print(df_name.tail(1))


# df_display(df)
# pint(df.dtypes)


# Splitting the DataFrame into 80% for df_sheetJ_80 and 20% for df_sheetJ_20
df_80, df_20 = train_test_split(df, test_size=0.2, random_state=42)
df_20.to_excel('CPTO_reduce_100.xlsx', index=False)
# Convert DataFrame to list of dictionaries
df_80_dict = df_80.to_dict(orient='records')
df_20_dict = df_20.to_dict(orient='records')

column_names = ['WORKSHOP', 'EQUIPMENT_GROUP', 'EQUIPMENT_SUB_GROUP', 'EQUIPMENT', 'PRODUCT',
                'WAFERS_COUNT', 'ALARM_DURING_PROCESS']

# Convert df into dict
df_20_dict_modified = []
for row_dict in df_20_dict:
    modified_row_dict = {key: row_dict[key] for key in column_names if key in row_dict}
    df_20_dict_modified.append(modified_row_dict)


# print(df_20_dict_modified)


def cal_range_multi(df, column_names):
    """
    Calculate the range of values for multiple columns in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the columns.
        column_names (list): List of column names for which to calculate the range.

    Returns:
        dict: A dictionary where keys are range_column_name and values are the corresponding column names.
    """
    ranges = {}
    for col_name in column_names:
        # print(df[col_name].max())
        # print(df[col_name].min())
        column_range = round(df[col_name].max() - df[col_name].min(), 1)
        # print(column_range)
        ranges[col_name] = column_range
    return ranges


ranges_dict = cal_range_multi(df_80, column_names)
print("Range: ", ranges_dict)

weight_set = [1, 1, 1, 1, 1, 1, 1]


def weight(df, column_names, weight_set):
    weights = {}
    for i, col_name in enumerate(column_names):
        weights[col_name] = weight_set[i]
    return weights


weights_dict = weight(df_80, column_names, weight_set)

print("Weight: ", weights_dict)


def calculate_similarity(df_list, ranges_dict, weights_dict, weight_set, column_names, new_value):
    similarities = []

    for row in range(len(df_list)):
        ls_line = {}
        glo_sumproduct = 0

        for col in column_names:
            local_similarity = round(1 - (abs(new_value[col] - df_list[row][col]) / ranges_dict[col]), 2)
            ls_line[col] = local_similarity
            glo_sumproduct += weights_dict[col] * local_similarity

        glo_similarity = round(((1 / sum(weight_set)) * glo_sumproduct), 2)

        similarities.append((row, ls_line, glo_similarity))

        # Sort similarities based on global similarity (GS)
    similarities_sorted = sorted(similarities, key=lambda x: x[2], reverse=True)

    # Get the top 1 row
    top_row_index, top_row_ls_line, top_row_glo_similarity = similarities_sorted[0]
    # Get last item
    last_item_key, last_item_value = list(df_80_dict[top_row_index].items())[-1]

    # Add the last item to the new_value_1 dictionary using the same key
    new_value[last_item_key] = last_item_value

    return top_row_index, top_row_ls_line, top_row_glo_similarity, new_value, last_item_value


df_20_results = []

# Loop through each row in df_sheetJ_20_dict_modified
for row in df_20_dict_modified:
    # Calculate similarity for the current new_value
    top_row_index, top_row_ls_line, top_row_glo_similarity, new_value, last_value = calculate_similarity(
        df_80_dict,
        ranges_dict,
        weights_dict, weight_set,
        column_names, row)

    # Store the results in a dictionary
    result = {
        "Top Row Index": top_row_index,
        "LS": top_row_ls_line,
        "GS": top_row_glo_similarity,
        "Add Temp": new_value,
        "last": last_value
    }

    # Append the result to the list
    df_20_results.append(result)

# Print the results
# for idx, result in enumerate(df_sheetJ_20_resusemantic stufflts, start=1):
#     print(f"Result {idx}: {result}")

excel_file_path = 'df_cpto_20_results_100.xlsx'
df_20_results = pd.DataFrame(df_20_results)
df_20_results.to_excel(excel_file_path, index=False)