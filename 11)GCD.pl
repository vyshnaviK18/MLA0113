% Base case: GCD of a number with itself is the number.
gcd(X, X, X).

% Recursive case: GCD of two numbers (X and Y) is the same as the GCD of X and (Y-X).
gcd(X, Y, Result) :-
    X < Y,
    Z is Y - X,
    gcd(X, Z, Result).

gcd(X, Y, Result) :-
    X > Y,
    gcd(Y, X, Result).

% Example usage:
% ?- gcd(24, 36, G).
% G = 12.

