import subprocess
from pysat.formula import WCNF

# Hàm ghi file WCNF với prefix 'h' cho hard clause
def write_wcnf_with_h_prefix(wcnf, filename):
    with open(filename, 'w') as f:
        for clause in wcnf.hard:
            f.write("h " + " ".join(map(str, clause)) + " 0\n")
        for item in wcnf.soft:
            if isinstance(item, tuple):
                clause, weight = item
            else:
                clause = item
                weight = 1
            f.write(f"{weight} " + " ".join(map(str, clause)) + " 0\n")

# Hàm gọi solver MaxHS
def call_maxhs_solver(wcnf_file):
    try:
        result = subprocess.run([
    'wsl', './MaxHS/build/release/bin/maxhs', '-printSoln',
    wcnf_file
], capture_output=True, text=True, timeout=3600)

        # Phân tích kết quả
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith('v '):
                print(line)
        return None
    except subprocess.TimeoutExpired:
        print("Solver timeout")
        return None

# Hàm chính
def main():
    # Tạo bài toán MaxSAT đơn giản
    wcnf = WCNF()

    # Hard clause: (x1 ∨ ¬x2)
    wcnf.hard.append([1, -2])

    # Soft clause: (x3 ∨ ¬x4) trọng số 5
    wcnf.soft.append(([3, -4], 5))

    # Soft clause: (¬x1 ∨ x4) trọng số 2
    wcnf.soft.append(([-1, 4], 2))

    # Ghi ra file WCNF
    filename = "problem.wcnf"
    #write_wcnf_with_h_prefix(wcnf, filename)

    # Gọi solver MaxHS
    call_maxhs_solver(filename)

if __name__ == "__main__":
    main()
