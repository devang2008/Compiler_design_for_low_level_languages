int fibonacci(int n) {
    if (n < 2) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
int main() {
    int i;
    for (i = 0; i < 10; i++) {
        print_int(fibonacci(i));
        print_char('\n');
    }
    return 0;
}
