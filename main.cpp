#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <pthread.h>

using namespace std;
using Matrix = vector<vector<int>>;

const int STRASSEN_THRESHOLD = 64;
int maxThreads = 4;  // задаётся из main()

// Прототипы
Matrix standardMultiply(const Matrix& A, const Matrix& B);
Matrix strassenSequential(const Matrix& A, const Matrix& B);
Matrix strassenParallelRecursive(const Matrix& A, const Matrix& B, int threadsLeft);
Matrix strassenParallel(const Matrix& A, const Matrix& B);
bool areMatricesEqual(const Matrix& A, const Matrix& B);

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

// Последовательный Strassen
Matrix strassenSequential(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B);

    int m = n / 2;
    // Разбиение на 4 подматрицы
    Matrix A11(m, vector<int>(m)), A12(m, vector<int>(m)),
           A21(m, vector<int>(m)), A22(m, vector<int>(m));
    Matrix B11(m, vector<int>(m)), B12(m, vector<int>(m)),
           B21(m, vector<int>(m)), B22(m, vector<int>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + m];
            A21[i][j] = A[i + m][j];
            A22[i][j] = A[i + m][j + m];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + m];
            B21[i][j] = B[i + m][j];
            B22[i][j] = B[i + m][j + m];
        }

    // 7 продуктов
    Matrix M1 = strassenSequential(add(A11, A22), add(B11, B22));
    Matrix M2 = strassenSequential(add(A21, A22), B11);
    Matrix M3 = strassenSequential(A11, subtract(B12, B22));
    Matrix M4 = strassenSequential(A22, subtract(B21, B11));
    Matrix M5 = strassenSequential(add(A11, A12), B22);
    Matrix M6 = strassenSequential(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassenSequential(subtract(A12, A22), add(B21, B22));

    // Сборка C
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            C[i][j]         = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + m]     = M3[i][j] + M5[i][j];
            C[i + m][j]     = M2[i][j] + M4[i][j];
            C[i + m][j + m] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    return C;
}

// Аргументы для потокового параллельного вызова
struct ParArgs {
    const Matrix* A;
    const Matrix* B;
    Matrix* result;
    int threadsLeft;
};

// Функция-обёртка для запуска strassenParallelRecursive в потоке
void* parWorker(void* arg) {
    auto* p = static_cast<ParArgs*>(arg);
    *p->result = strassenParallelRecursive(*p->A, *p->B, p->threadsLeft);
    return nullptr;
}

// Рекурсивный параллельный Strassen на всех уровнях
Matrix strassenParallelRecursive(const Matrix& A, const Matrix& B, int threadsLeft) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD || threadsLeft <= 1)
        return strassenSequential(A, B);

    int m = n / 2;
    // Разбиение
    Matrix A11(m, vector<int>(m)), A12(m, vector<int>(m)),
           A21(m, vector<int>(m)), A22(m, vector<int>(m));
    Matrix B11(m, vector<int>(m)), B12(m, vector<int>(m)),
           B21(m, vector<int>(m)), B22(m, vector<int>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + m];
            A21[i][j] = A[i + m][j];
            A22[i][j] = A[i + m][j + m];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + m];
            B21[i][j] = B[i + m][j];
            B22[i][j] = B[i + m][j + m];
        }

    // Подготовка семи пар (A, B)
    Matrix As[7], Bs[7], Ms[7];
    As[0] = add(A11, A22);       Bs[0] = add(B11, B22);
    As[1] = add(A21, A22);       Bs[1] = B11;
    As[2] = A11;                 Bs[2] = subtract(B12, B22);
    As[3] = A22;                 Bs[3] = subtract(B21, B11);
    As[4] = add(A11, A12);       Bs[4] = B22;
    As[5] = subtract(A21, A11);  Bs[5] = add(B11, B12);
    As[6] = subtract(A12, A22);  Bs[6] = add(B21, B22);

    // Сколько потоков создаём (не больше 7 и не больше threadsLeft‑1)
    int spawn = min(threadsLeft - 1, 7);
    pthread_t tids[7];
    ParArgs args[7];

    // Запускаем первые spawn задач в отдельных потоках
    for (int i = 0; i < spawn; i++) {
        args[i] = { &As[i], &Bs[i], &Ms[i], threadsLeft - 1 };
        pthread_create(&tids[i], nullptr, parWorker, &args[i]);
    }
    // Оставшиеся задачи выполняем в текущем потоке (но тоже рекурсивно параллельно)
    for (int i = spawn; i < 7; i++) {
        Ms[i] = strassenParallelRecursive(As[i], Bs[i], threadsLeft - 1);
    }
    // Ждём завершения потоков
    for (int i = 0; i < spawn; i++) {
        pthread_join(tids[i], nullptr);
    }

    // Сборка итоговой матрицы
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            C[i][j]         = Ms[0][i][j] + Ms[3][i][j]
                            - Ms[4][i][j] + Ms[6][i][j];
            C[i][j + m]     = Ms[2][i][j] + Ms[4][i][j];
            C[i + m][j]     = Ms[1][i][j] + Ms[3][i][j];
            C[i + m][j + m] = Ms[0][i][j] - Ms[1][i][j]
                            + Ms[2][i][j] + Ms[5][i][j];
        }
    return C;
}

Matrix strassenParallel(const Matrix& A, const Matrix& B) {
    return strassenParallelRecursive(A, B, maxThreads);
}

bool areMatricesEqual(const Matrix& A, const Matrix& B) {
    int n = A.size();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (A[i][j] != B[i][j])
                return false;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        maxThreads = stoi(argv[1]);
        if (maxThreads < 1) maxThreads = 1;
    }
    cout << "Using up to " << maxThreads << " threads\n";

    int n = 2048;
    Matrix A = generateRandomMatrix(n);
    Matrix B = generateRandomMatrix(n);

    Matrix C_std, C_seq, C_par;
    double t_std, t_seq, t_par;

    auto t0 = chrono::high_resolution_clock::now();
    C_std = standardMultiply(A, B);
    auto t1 = chrono::high_resolution_clock::now();
    t_std = chrono::duration<double, milli>(t1 - t0).count();

    auto t2 = chrono::high_resolution_clock::now();
    C_seq = strassenSequential(A, B);
    auto t3 = chrono::high_resolution_clock::now();
    t_seq = chrono::duration<double, milli>(t3 - t2).count();

    auto t4 = chrono::high_resolution_clock::now();
    C_par = strassenParallel(A, B);
    auto t5 = chrono::high_resolution_clock::now();
    t_par = chrono::duration<double, milli>(t5 - t4).count();

    cout << "Standard multiply:   " << t_std << " ms\n";
    cout << "Sequential Strassen: " << t_seq << " ms\n";
    cout << "Parallel Strassen:   " << t_par << " ms\n\n";

    cout << "Std vs Seq: " << (areMatricesEqual(C_std, C_seq) ? "Equal\n" : "Not equal\n");
    cout << "Std vs Par: " << (areMatricesEqual(C_std, C_par) ? "Equal\n" : "Not equal\n");
    cout << "Seq vs Par: " << (areMatricesEqual(C_seq, C_par) ? "Equal\n" : "Not equal\n");

    return 0;
}
