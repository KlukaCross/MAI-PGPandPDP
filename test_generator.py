#! /usr/bin/env python3

import random
import sys

RAND_MIN = -1000
RAND_MAX = 1000


def gen_random_vectors(n: int) -> tuple[list[float], list[float]]:
    def random_vector():
        return [random.randint(RAND_MIN, RAND_MAX)*random.random() for _ in range(n)]
    return random_vector(), random_vector()


def get_answer(a: list[float], b: list[float]):
    return [a[i]-b[i] for i in range(len(a))]


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <test dir> <count for tests>")
        sys.exit(1)

    test_dir = sys.argv[1]
    count_of_tests = int(sys.argv[2])

    for i in range(count_of_tests):
        n = 10**i
        a, b = gen_random_vectors(n)
        answer = get_answer(a, b)

        test_file_name = f"{test_dir}/{i+1:02d}"
        with open("{0}.t".format(test_file_name), 'w') as output_file, \
                open("{0}.a".format(test_file_name), "w") as answer_file:
            output_file.write(f"{n}\n{' '.join(map(str, a))}\n{' '.join(map(str, b))}\n")
            answer_file.write(f"{' '.join(map(str, answer))}\n")


main()
