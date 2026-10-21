from tools import (
    get_deployed_model,
    get_model_inputs,
    get_feature_data
)


model = get_deployed_model()

print("\nDEPLOYED MODEL:")
print(model)

features = get_model_inputs(model["version_id"])

print("\nFETCHING FIRST FEATURE DATA...")

feature = features[0]

# Extract slice number from feature name
slice_id = 1 if feature["name"].startswith("slice1") else 5

# Extract source column
source_column = feature["source_expr"].split(".")[-1]

data = get_feature_data(
    run_id=model["run_id"],
    slice_id=slice_id,
    source_column=source_column,
    limit=10
)

print("\nFEATURE:")
print(feature["name"])

print("\nSOURCE COLUMN:")
print(source_column)

print("\nDATA:")
for row in data:
    print(row)