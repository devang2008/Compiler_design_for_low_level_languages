int main() {
    int arr[5];
    int i;
    for (i = 0; i < 5; i++) {
        arr[i] = i * 2;
    }
    for (i = 0; i < 5; i++) {
        print_int(arr[i]);
        print_char('\n');
    }
    return 0;
}
