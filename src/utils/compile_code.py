import os
import argparse

def compile_scripts_to_txt(directory, output_filename="compiled_code.txt"):
    # Get all files in the specified directory
    files = os.listdir(directory)
    
    # Filter to include only specific code file types
    code_files = [f for f in files if f.endswith(('.py', '.sh', '.txt', '.cpp', '.c', '.js', '.java'))]
    
    # Open the output file in write mode
    with open(output_filename, 'w') as outfile:
        for file_name in code_files:
            # Add a header for each file
            outfile.write(f"\n\n# {'=' * 10} {file_name} {'=' * 10}\n\n")
            
            # Read and write each file's content
            with open(os.path.join(directory, file_name), 'r') as infile:
                outfile.write(infile.read())
    
    print(f"Compiled {len(code_files)} code files from '{directory}' into '{output_filename}'.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compile code scripts from a specified directory into a single text file.")
    parser.add_argument("--dir", type=str, help="Directory to compile code scripts from")
    parser.add_argument("--output", type=str, default="compiled_code.txt", help="Output filename for the compiled text (default: compiled_code.txt)")

    args = parser.parse_args()
    
    # Run the compile function with provided arguments
    compile_scripts_to_txt(args.dir, args.output)
