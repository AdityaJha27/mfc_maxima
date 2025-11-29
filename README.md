# mfc_maxima
## Q1. Create and transform vectors and matrices (the transpose vector (matrix) conjugate
#### a. transpose of a vector (matrix)
    /* Horizontal complex vector */
    h: [1 + 2*%i, 2 - %i, 3.0];
    print("Horizontal vector h: ", h);
    
    /* Vertical (column) vector */
    v: transpose(h);
    print("Vertical vector v (column): ", v);
    
    /* Matrix creation */
    A: matrix([1, 2 + %i, 3],
              [4, 5, 6 - 2*%i],
              [7, 8, 9]);
    print("Matrix A: ", A);
    
    /* Transpose */
    At: transpose(A);
    print("Transpose A^T: ", At);
    
    /* Conjugate transpose (Hermitian) */
    A_H: conjugate(transpose(A));
    print("Conjugate Transpose (A^H): ", A_H);

## Q2. Generate the matrix into echelon form and find its rank.
    /* Matrix creation */
    M: matrix([1, 2, 0],
              [2, 4, 1],
              [3, 6, 3]);
    print("Matrix M: ", M);
    
    /* Row Echelon Form (REF) */
    REF_M: echelon(M);
    print("Echelon Form of M: ", REF_M);
    
    /* Reduced Row Echelon Form (RREF) */
    RREF_M: rref(M);
    print("Reduced Row Echelon Form (RREF): ", RREF_M);
    
    /* Rank of Matrix */
    rank_M: rank(M);
    print("Rank of M: ", rank_M);

## Q3. Find cofactors, determinant, adjoint and inverse of a matrix.
    /* Matrix Creation */
    B: matrix([2, 3, 1],
              [1, 0, 4],
              [5, 2, 2]);
    print("Matrix B: ", B);
    
    /* Determinant */
    detB: determinant(B);
    print("det(B) = ", detB);
    
    /* Cofactor Matrix */
    cofactorB: cofactors(B);
    print("Cofactor Matrix: ", cofactorB);
    
    /* Adjoint (Transpose of Cofactor Matrix) */
    adjointB: transpose(cofactorB);
    print("Adjoint of B: ", adjointB);
    
    /* Inverse of B (only if determinant non-zero) */
    if detB # 0 then (
        invB: invert(B),
        print("Inverse B^-1: ", invB)
    )
    else (
        print("B is singular, inverse does not exist.")
    );

## Q4. Solve a system of Homogeneous and non-homogeneous equations using Gauss
#### a. elimination method.
    /* Non-Homogeneous System: Ax = b */
    A: matrix([2, -1, 1],
              [3, 2, -4],
              [1, 1, 1]);
    
    b: matrix([1], [-2], [6]);  /* Column vector b */
    
    print("Matrix A: ", A);
    print("Vector b: ", b);
    
    /* Solve Ax = b using Gaussian elimination (linsolve) */
    x: linsolve(A, b);
    print("Solution x : ", x);
    
    /* Homogeneous System Ax = 0 */
    A2: matrix([1, 2, 3],
               [2, 4, 6],
               [1, 1, 1]);
    
    print("Matrix A2: ", A2);
    
    /* Solve A2 * x = 0 (Nullspace) */
    nullA2: nullspace(A2);
    print("Nullspace basis vectors (solution of A2*x = 0): ", nullA2);


## Q5. Solve a system of Homogeneous equations using the Gauss Jordan method.
    /* Homogeneous System Matrix C */
    C: matrix([1, 2, -1, 0],
              [2, 4, -2, 0],
              [3, 6, -3, 0]);
    print("Matrix C: ", C);
    
    /* Gauss-Jordan: Reduced Row Echelon Form */
    RREF_C: rref(C);
    print("RREF of C: ", RREF_C);
    
    /* Nullspace → Basis of solutions of Cx = 0 */
    null_C: nullspace(C);
    print("Nullspace basis (solutions of Cx = 0): ", null_C);

## Q6. Generate basis of column space, null space, row space and left null space of a matrix space.
    /* Matrix Creation */
    D: matrix([1, 2, 3],
              [2, 4, 6],
              [1, 0, 1]);
    print("Matrix D: ", D);
    
    /* Column Space Basis */
    col_D: columnspace(D);
    print("Column space basis: ", col_D);
    
    /* Row Space Basis */
    row_D: rowspace(D);
    print("Row space basis: ", row_D);
    
    /* Null Space Basis (solutions of Dx = 0) */
    null_D: nullspace(D);
    print("Null space basis: ", null_D);
    
    /* Left Null Space Basis (nullspace of D^T) */
    left_null_D: nullspace(transpose(D));
    print("Left null space basis: ", left_null_D);

## Q7. Check the linear dependence of vectors. Generate a linear combination of given vectors of R^n/ matrices of the same size and find the transition matrix of given matrix space.
    /* Vectors in R^3 */
    v1: matrix([1], [2], [3]);
    v2: matrix([2], [4], [6]);
    v3: matrix([0], [1], [1]);
    
    /* Matrix with vectors as columns */
    M: matrix(
            [1, 2, 0],
            [2, 4, 1],
            [3, 6, 1]
         );
    print("Matrix M: ", M);
    
    /* Linear Dependence Check */
    r: rank(M);
    print("Rank of M: ", r);
    print("Number of vectors: ", 3);
    
    if r < 3 then
        print("=> Vectors are linearly dependent")
    else
        print("=> Vectors are linearly independent");
    
    /* Linear Combination: c1*v1 + c2*v2 + c3*v3 */
    depends([c1, c2, c3], true);   /* Declare as symbols */
    lin_comb: c1*v1 + c2*v2 + c3*v3;
    print("Linear Combination c1*v1 + c2*v2 + c3*v3 = ", lin_comb);
    
    /* Transition Matrix Construction */
    e1: matrix([1], [0], [0]);  /* Standard basis vector */
    
    B: matrix(
            [1, 0, 1],
            [2, 1, 0],
            [3, 1, 0]
         );
    print("Basis Matrix B: ", B);
    
    /* Change of basis matrix P (B → Standard) */
    P: B;
    print("Change of basis matrix P: ", P);
    
    /* Inverse matrix (Standard → B basis) */
    P_inv: invert(P);
    print("Inverse of P (Standard → B): ", P_inv);

## Q8. Find the orthonormal basis of a given vector space using the Gram-Schmidt orthogonalization process.
    /* Given vectors */
    v1: [1, 1, 0];
    v2: [1, 0, 1];
    v3: [0, 1, 1];
    
    M: matrix(v1, v2, v3);
    print("Original vectors (as rows shown): ", M);
    
    /* Transpose needed so that vectors become columns */
    Mt: transpose(M);
    print("Vectors as columns (Mt): ", Mt);
    
    /* QR Decomposition: Mt = Q * R */
    Q_R: qr(Mt);
    Q: Q_R[1];
    R: Q_R[2];
    
    print("Orthogonal Q matrix: ", Q);
    
    /* Orthonormal Basis = columns of Q */
    print("Orthonormal basis vectors:");
    b1: Q[1];
    b2: Q[2];
    b3: Q[3];
    print("b1 = ", b1);
    print("b2 = ", b2);
    print("b3 = ", b3);
    
    /* Check norms for verification */
    print("Norm of b1: ", sqrt(b1.b1));
    print("Norm of b2: ", sqrt(b2.b2));
    print("Norm of b3: ", sqrt(b3.b3));

## Q9. Check the diagonalizable property of matrices and find the corresponding eigenvalue and verify the Cayley-Hamilton theorem.

    /* Matrix E */
    E: matrix([5, 4, 2],
              [0, 1, 0],
              [0, 0, 3]);
    print("Matrix E:", E);
    
    /* Eigenvalues and Eigenvectors */
    ev: eigenvectors(E);
    print("Eigenvalues and eigenvectors: ", ev);
    
    /* Check Diagonalizable */
    diag_check: diagonalizable(E);
    print("Is matrix diagonalizable? ", diag_check);
    
    /* If diagonalizable: find P and D */
    if diag_check then (
        P: ev[2],    /* Matrix of eigenvectors */
        D: ev[1],    /* Diagonal matrix of eigenvalues */
        print("Matrix P (eigenvectors): ", P),
        print("Diagonal matrix D: ", D)
    );
    
    /* Cayley-Hamilton Theorem */
    charpoly: charpoly(E, x);
    print("Characteristic polynomial p(x): ", charpoly);
    
    /* Substitute E in p(x) → p(E) */
    pE: ev(E);  /* Cayley-Hamilton validation */
    
    print("p(E) should be Zero Matrix:");
    print(pE);
    
## Q10. Application of Linear algebra: Coding and decoding of messages using nonsingular matrices. e.g., code “Linear Algebra is fun” and then decode it.
    /* Q10: Coding and Decoding using Linear Algebra (Hill Cipher) */
    
    /* Define the Key Matrix (must be invertible modulo 26) */
    K : matrix([3, 3], [2, 5]);
    
    /* Function: Convert character to number (A=0 ... Z=25) */
    char_to_num(ch) := charcode(ch) - charcode("A");
    
    /* Function: Convert number to character */
    num_to_char(n) := string(char(n + charcode("A")));
    
    /* Convert message to numeric list */
    message : "LINEARALGEBRAISFUN"$
    msg_list : []$
    for i:1 thru string_length(message) do
        msg_list : endcons(char_to_num(charat(message, i)), msg_list)$
    
    /* Pad to make even number of characters */
    if remainder(length(msg_list), 2) # 0 then
        msg_list : endcons(char_to_num("X"), msg_list)$
    
    /* Encoding */
    encoded : []$
    for i:1 thru length(msg_list) step 2 do (
        block : matrix([msg_list[i]], [msg_list[i+1]]),
        cipher : rest(matrixmul(K, block), 2) mod 26,
        encoded : append(encoded, flatten(cipher))
    )$
    
    print("Encoded Numeric: ", encoded);
    
    /* Convert numbers to ciphertext letters */
    ciphertext : ""$
    for n in encoded do
        ciphertext : concat(ciphertext, num_to_char(n))$
    
    print("Ciphertext: ", ciphertext);
    
    /* ----- Decoding part ----- */
    
    /* Compute inverse key modulo 26 */
    detK : determinant(K);
    inv_detK : inverse_mod(detK, 26);
    adjK : adjoint(K);
    K_inv_mod26 : (inv_detK * adjK) mod 26;
    
    /* Decoding */
    decoded : []$
    for i:1 thru length(encoded) step 2 do (
        block2 : matrix([encoded[i]], [encoded[i+1]]),
        plain : rest(matrixmul(K_inv_mod26, block2), 2) mod 26,
        decoded : append(decoded, flatten(plain))
    )$
    
    print("Decoded Numeric: ", decoded);
    
    decoded_text : ""$
    for n in decoded do
        decoded_text : concat(decoded_text, num_to_char(n))$
    
    print("Decoded Text: ", decoded_text);
    
## Q11. Compute Gradient of a scalar field
    /* Q11: Gradient of a scalar field */
    
    /* Define variables */
    x : 'x;
    y : 'y;
    z : 'z;
    
    /* Define scalar field f(x,y,z) */
    f : x^2 * y + sin(y*z) + exp(x*z);
    print("Scalar field f(x,y,z): ", f);
    
    /* Compute gradient ∇f */
    grad_f : [diff(f, x), diff(f, y), diff(f, z)];
    print("Gradient ∇f: ", grad_f);
    
    /* Optional: Display as column vector */
    grad_vec : matrix(grad_f);
    print("Gradient vector ∇f (column form): ", grad_vec);

## Q12. Compute Divergence of a vector field.
    /* Q12: Divergence of a Vector Field */
    
    /* Define variables */
    x : 'x;
    y : 'y;
    z : 'z;
    
    /* Define vector field components */
    P : x*y^2;
    Q : sin(x*z);
    R : exp(y*z);
    
    print("Vector field F = (P, Q, R): ", [P, Q, R]);
    
    /* Compute divergence */
    divF : diff(P, x) + diff(Q, y) + diff(R, z);
    divF_simplified : ratsimp(divF);  /* Simplify expression */
    
    print("Divergence ∇·F = ", divF_simplified);

## Q13. Compute Curl of a vector field.
    /* Q13: Curl of a Vector Field */
    
    /* Define variables */
    x : 'x;
    y : 'y;
    z : 'z;
    
    /* Vector field components */
    P : y*z;
    Q : x*z;
    R : x*y;
    
    print("Vector field F = (P, Q, R): ", [P, Q, R]);
    
    /* Compute curl components */
    curl_x : diff(R, y) - diff(Q, z);
    curl_y : diff(P, z) - diff(R, x);
    curl_z : diff(Q, x) - diff(P, y);
    
    /* Curl vector as column */
    curl_vec : matrix([curl_x], [curl_y], [curl_z]);
    
    print("Curl ∇×F = ", curl_vec);






