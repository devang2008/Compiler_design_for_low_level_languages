int main() {
    int x;
    int sum;
    sum = 0;
    x = 1;
    while (x <= 10) {
        sum = sum + x;
        x++;
    }
    print_int(sum);
    return 0;
}
