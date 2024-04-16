# extract.py
import argparse
import pickle

def extract_betas(pkl_path):
    # Load the .pkl file to access its contents
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    # Extract the "betas" key from the data
    betas = data.get('betas', None)

    # Check if "betas" was found
    if betas is not None:
        # Convert the betas numpy array to a list for text representation
        betas_list = betas.tolist()
        return betas_list
    else:
        print("Betas key not found")
        return None

if __name__ == "__main__":
    # Set up argument parsing only if this script is run as the main program
    parser = argparse.ArgumentParser(description='Extract information from a .pkl file.')
    parser.add_argument('pkl_path', type=str, help='Path to the .pkl file to be processed.')
    args = parser.parse_args()

    # Call the extract_betas function and print the result
    betas_list = extract_betas(args.pkl_path)
