#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <pthread.h>
#include <semaphore.h>

using namespace std;
using Matrix = vector<vector<int>>;

const int STRASSEN_THRESHOLD = 64;
int MAX_THREADS = 1;
sem_t thread_sem;

struct ThreadData {
    Matrix A;
    Matrix B;
    Matrix result;
};

Matrix generateRandomMatrix(int n) {
    Matrix A(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = rand() % 10;
    return A;
}

Matrix standardMultiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

Matrix add(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix subtract(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

Matrix strassenSequential(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B);

    int newSize = n / 2;
    Matrix A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)),
           A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)),
           B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize));

    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }

    Matrix M1 = strassenSequential(add(A11, A22), add(B11, B22));
    Matrix M2 = strassenSequential(add(A21, A22), B11);
    Matrix M3 = strassenSequential(A11, subtract(B12, B22));
    Matrix M4 = strassenSequential(A22, subtract(B21, B11));
    Matrix M5 = strassenSequential(add(A11, A12), B22);
    Matrix M6 = strassenSequential(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassenSequential(subtract(A12, A22), add(B21, B22));

    Matrix C(n, vector<int>(n));
    for (int i = 0; i < newSize; ++i)
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + newSize] = M3[i][j] + M5[i][j];
            C[i + newSize][j] = M2[i][j] + M4[i][j];
            C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }

    return C;
}

Matrix strassenPOSIX(const Matrix& A, const Matrix& B);

void* computeMParallel(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    data->result = strassenPOSIX(data->A, data->B);
    sem_post(&thread_sem);
    return nullptr;
}

Matrix strassenPOSIX(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B);

    int newSize = n / 2;
    Matrix A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)),
           A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)),
           B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize));

    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }

    ThreadData M_data[7];
    pthread_t threads[7];
    bool thread_created[7] = {false};

    M_data[0].A = add(A11, A22); M_data[0].B = add(B11, B22);
    M_data[1].A = add(A21, A22); M_data[1].B = B11;
    M_data[2].A = A11;           M_data[2].B = subtract(B12, B22);
    M_data[3].A = A22;           M_data[3].B = subtract(B21, B11);
    M_data[4].A = add(A11, A12); M_data[4].B = B22;
    M_data[5].A = subtract(A21, A11); M_data[5].B = add(B11, B12);
    M_data[6].A = subtract(A12, A22); M_data[6].B = add(B21, B22);

    for (int i = 0; i < 7; ++i) {
        if (sem_trywait(&thread_sem) == 0) {
            pthread_create(&threads[i], nullptr, computeMParallel, &M_data[i]);
            thread_created[i] = true;
        } else {
            M_data[i].result = strassenPOSIX(M_data[i].A, M_data[i].B);
        }
    }

    for (int i = 0; i < 7; ++i) {
        if (thread_created[i]) {
            pthread_join(threads[i], nullptr);
        }
    }

    Matrix C(n, vector<int>(n));
    for (int i = 0; i < newSize; ++i)
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = M_data[0].result[i][j] + M_data[3].result[i][j] - 
                      M_data[4].result[i][j] + M_data[6].result[i][j];
            C[i][j + newSize] = M_data[2].result[i][j] + M_data[4].result[i][j];
            C[i + newSize][j] = M_data[1].result[i][j] + M_data[3].result[i][j];
            C[i + newSize][j + newSize] = M_data[0].result[i][j] - M_data[1].result[i][j] + 
                                          M_data[2].result[i][j] + M_data[5].result[i][j];
        }

    return C;
}

bool validateMatrices(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n != B.size()) return false;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (A[i][j] != B[i][j]) return false;
    return true;
}

int main(int argc, char* argv[]) {
    int n = 1024;
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) MAX_THREADS = atoi(argv[2]);

    sem_init(&thread_sem, 0, MAX_THREADS);

    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Max threads: " << MAX_THREADS << endl;

    Matrix A = generateRandomMatrix(n);
    Matrix B = generateRandomMatrix(n);

    auto start = chrono::high_resolution_clock::now();
    Matrix C_std = standardMultiply(A, B);
    auto end = chrono::high_resolution_clock::now();
    cout << "Standard: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    Matrix C_seq = strassenSequential(A, B);
    end = chrono::high_resolution_clock::now();
    cout << "Strassen Sequential: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    Matrix C_par = strassenPOSIX(A, B);
    end = chrono::high_resolution_clock::now();
    cout << "Strassen POSIX: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

    cout << "Validation (Sequential): " << (validateMatrices(C_std, C_seq) ? "Passed" : "Failed") << endl;
    cout << "Validation (Parallel):   " << (validateMatrices(C_std, C_par) ? "Passed" : "Failed") << endl;

    sem_destroy(&thread_sem);
    return 0;
}