//
// Created by jerry on 2022/5/12.
//
#include "include/xmap_util.hpp"

int main(int argc, char **argv) {
    arma::mat A = {{1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
    };

    arma::mat B = {{1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5},
    };

    printf("[%llu, %llu]\n", A.n_rows, A.n_cols);
    A.print("A:");
    arma::uvec idx = arma::find(A > 3 && A < 5);
    idx.print("idx:");

    auto C = A;
    C.elem(idx).fill(0);
    C.print("C:");

    arma::vec V = arma::vectorise(A);
    V.print("V:");
    printf("V shape [%llu, %llu]\n", V.n_rows, V.n_cols);

    auto W = arma::reshape(V, A.n_rows, A.n_cols);
    W.print("W:");

    auto M = arma::mean(A);
    M.print("M:");

    auto D = A % B; // 矩阵乘，逐元素
    D.print("D:");

    auto E = A * 3;
    E.print("E:");
    // numpy : A[1:-1, 1:-1]
    auto H = A.submat(1, 1, A.n_rows - 1, A.n_cols - 1);
    H.print("H");
    // numpy: local_mean_center[1:-1, 1:-1] - np.array([1, 2])
    arma::cube Z(6, 5, 2);
    Z.slice(0) = A - 1;
    Z.slice(1) = B - 2;
    Z.print("Z:");
    arma::ucube z_max = arma::index_max(Z, 2);
    z_max.print("z_max:");
    arma::uvec z_idx = arma::find(Z > 0, 255, "first");
    z_idx.print("z_idx");
    printf("------------------------------------------\n");

    arma::cube Y;
    Y = arma::join_slices(A, B);
    Y = arma::join_slices(Y, B);
    Y.print("Y:");
    Y.tube(0, 0, 0, 0).print("Tube00:");
    arma::cube U = arma::reshape(Y, 30, 3, 1);
    U.slice(0).print("U:");

    arma::vec c1 = {1, 2, 3, 4, 5, 6};
    arma::vec c2 = {6, 5, 4, 3, 2, 1};
    arma::vec c3 = {6, 5, 4, 3, 2, 1};
    arma::mat O = arma::join_rows(c1, c2, c3);
    O.print("O:");


    arma::mat A1 = {{1, 2, 3},
                    {2, 2, 3},
    };
    arma::mat B1 = {{1, 2},
                    {1, 2},
                    {1, 2},
    };
//    arma::mat t = A1.t();
    arma::mat P = A1 * A1.t() * A1;
    P.print("P:");

    arma::mat G = arma::inv(A1 * A1.t()) * (A1 * A1.t());
    G.print("G:");

    arma::mat A1_rep = arma::repmat(A1, 2, 3);
    A1_rep.print("A1_rep:");

    // numpy: np.transpose(np.array(np.meshgrid(t_y, t_x), dtype=np.uint64), axes=(2, 1, 0))  # [h, w, 2]
    arma::vec c4 = arma::linspace(0, 5, 6);
    arma::vec c5 = arma::linspace(0, 3, 4);
    arma::mat J = arma::reshape(c4, 6, 1);
    arma::mat J_rep = arma::repmat(J, 1, 4);
    J_rep.print("J_rep:");
    arma::mat J1 = arma::reshape(c5, 1, 4);
    arma::mat J1_rep = arma::repmat(J1, 6, 1);
    J1_rep.print("J1_rep:");

    arma::vec J1_X = arma::vectorise(J_rep);
    arma::vec J1_Y = arma::vectorise(J1_rep);
    J1_X.print("J1_X:");
    J1_Y.print("J1_Y:");

    arma::vec S = arma::vectorise(Z);
    double std = arma::stddev(S);
    printf("std: %f \n", std);

    arma::cube N = arma::join_slices(A1, A1);
    arma::vec col = {3.14, 2.68};
    N.tube(1, 2) = col;
    N.print("N:");

    arma::vec m1 = {1, 2, 3, 4, 5};
//    arma::vec m2 = {1, 2, 3, 4, 5};
    auto L = arma::pow(m1, 2);
    L.print("L:");

    arma::vec n1 = {1, 2, 3, 4, 5};
    arma::vec n2 = {5, 4, 3, 2, 1};
    arma::mat X = arma::join_rows(n1, n2);
    X.print("X:");

    arma::mat c(10, 3, arma::fill::randn);
    c.print("c:");
    c.row(0).print("c.row(0)");

    arma::cube aa(1, 1, 3, arma::fill::randn);
    arma::cube bb(10, 10, 3, arma::fill::randn);
    arma::cube ab = aa + bb;
    ab.print("ab:");
}