
def read_and_print_binary_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            w_bytes = file.read(4)
            h_bytes = file.read(4)

            w = int.from_bytes(w_bytes, byteorder='little')
            h = int.from_bytes(h_bytes, byteorder='little')

            print(f"Размер матрицы: {w} на {h}")

            for _ in range(h):
                row = []
                for _ in range(w):
                    element_bytes = file.read(4)
                    hex_str = element_bytes.hex().upper()
                    formatted_hex = ''.join([hex_str[i:i+2] for i in range(0, len(hex_str), 2)])
                    row.append(formatted_hex)
                print(" ".join(row))

    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    file_path = input("file path: ")
    read_and_print_binary_file(file_path)
