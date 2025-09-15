#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;
// --------------- Utility Functions ---------------

// Calculates the smallest odd grid size (≥ 3) that can hold the message
int calcAutoGrid(const string& msg) {
    int len = static_cast<int>(msg.size());
    for (int N = 3; ; N += 2) {
        int C = (N - 1) / 2;
        int capacity = 2 * C * (C + 1) + 1;
        if (capacity >= len) return N;
    }
}

// Processes raw input: removes spaces, converts to uppercase
string rawtext(const string& raw) {
    string msg;
    for (char ch : raw)
        if (!isspace(ch))
            msg.push_back(toupper(ch));
    return msg;
}

// Gets a valid integer input with exception handling
int getInt(const string& prompt) {
    while (true) {
        cout << prompt;
        string line;
        getline(cin, line);
        try { return stoi(line); }
        catch (...) { cout << "Enter a valid integer.\n"; }
    }
}

// Gets a string input
string getString(const string& prompt) {
    cout << prompt;
    string s;
    getline(cin, s);
    return s;
}

// -------------- Class: Grid -----------------
class Grid {
private:
    int size;
    vector<vector<char>> mat;
    mt19937 rng;
    uniform_int_distribution<int> uni;
    vector<pair<int,int>> coords;

    // Generates diamond shape
    void diamond() {
        int c = size / 2;
        for (int r = c; r >= 1; --r) {
            int i = c, j = c - r;
            for (int s = 0; s < r; ++s, --i, ++j) coords.push_back(make_pair(i, j));
            for (int s = 0; s < r; ++s, ++i, ++j) coords.push_back(make_pair(i, j));
            for (int s = 0; s < r; ++s, ++i, --j) coords.push_back(make_pair(i, j));
            for (int s = 0; s < r; ++s, --i, --j) coords.push_back(make_pair(i, j));
        }
        coords.push_back(make_pair(c, c));
    }

public:
    // Grid Constructor 
     Grid(int N)
      : size(N),
        mat(N, vector<char>(N, ' ')),
        rng(random_device{}()),
        uni('A', 'Z')
    {
        diamond();
    }

    // Fills grid with message following diamond shape
    void fillDiamond(const string& raw) {
        int idx = 0, L = static_cast<int>(raw.size());
        for (const auto& p : coords) {
            if (idx < L) mat[p.first][p.second] = raw[idx++];
            else break;
        }
    }

    // Fills remaining empty cells with random letters
    void fillRandom() {
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                if (mat[i][j] == ' ')
                    mat[i][j] = static_cast<char>(uni(rng));
    }

    // Reads grid column-wise to produce encrypted message
    string readColumn() const {
        string out; out.reserve(size * size);
        for (int c = 0; c < size; ++c)
            for (int r = 0; r < size; ++r)
                out.push_back(mat[r][c]);
        return out;
    }

    // Populates grid column-wise from ciphertext
    void populateColumn(const string& cipher) {
        int idx = 0;
        for (auto& row : mat) fill(row.begin(), row.end(), ' ');
        for (int c = 0; c < size && idx < (int)cipher.size(); ++c)
            for (int r = 0; r < size && idx < (int)cipher.size(); ++r)
                mat[r][c] = cipher[idx++];
    }

    // Reads grid in diamond pattern for decryption
    // Returns the decrypted message, stopping at a period if specified
    string readDiamondPattern(bool period) const {
        string out; out.reserve(coords.size());
        for (const auto& p : coords) {
            char ch = mat[p.first][p.second];
            if (ch == ' ') continue;
            out.push_back(ch);
            if (period && ch == '.') break;
        }
        return out;
    }

    // Prints grid 
    void print(const string& title = "") const {
        if (!title.empty()) cout << title << "\n";
        cout << "    ";
        for (int j = 0; j < size; ++j) cout << j << ' ';
        cout << "\n   " << string(2 * size + 1, '-') << "\n";
        for (int i = 0; i < size; ++i) {
            cout << setw(2) << i << "| ";
            for (int j = 0; j < size; ++j)
                cout << (mat[i][j] == ' ' ? '_' : mat[i][j]) << ' ';
            cout << "\n";
        }
        cout << "   " << string(2 * size + 1, '-') << "\n\n";
    }
};

// ------------------- Class: Encryptor ----------------------
class Encryptor {
public:
    // Performs one-round encryption with autimaitc or specified grid size
    string encryption(const string& raw, int gridSize, bool appendDot) {
        string msg = rawtext(raw);
        if (appendDot && !msg.empty() && msg.back() != '.')
            msg.push_back('.');
        Grid g(gridSize);
        g.fillDiamond(msg);
        g.print("*** After diamond fill ***");
        g.fillRandom();
        g.print("*** After random fill ***");
        return g.readColumn();
    }

    // Performs multi-round encryption with automatic grid sizes
    string multiEncryption(const string& raw, int rounds, vector<int>& gridSizes)
    {
        gridSizes.clear();
        string msg = raw;
    for (int i = 0; i < rounds; ++i) {
        cout << "\n=== Encryption Round " << (i + 1) << " ===\n";

        // Append dot before grid sizing
        if (i == 0 && !msg.empty() && msg.back() != '.')
            msg.push_back('.');

        int S = calcAutoGrid(msg);
        cout << "Auto-selected grid size: " << S << "\n";
        gridSizes.push_back(S);
        cout << "Using grid size: " << S
            << " for round " << (i + 1) << "\n";

        msg = encryption(msg, S, false);
        cout << "Encrypted after round " << (i + 1)
            << ": " << msg << "\n";
        }
        return msg;
    }
};

// ----------------------- Class: Decryptor -----------------------
class Decryptor {
public:
    // Performs one-round decryption
    string decrypt(const string& cipher,
                   int gridSize,
                   bool period)
    {
        Grid g(gridSize);
        g.populateColumn(cipher);
        g.print("*** Populated grid ***");
        return g.readDiamondPattern(period);
    }

    // Performs multi-round decryption using stored grid sizes
    string multiRoundDecrypt(const string& cipher,
                             int rounds,
                             const vector<int>& gridSizes)
    {
        string msg = cipher;
        for (int i = rounds - 1; i >= 0; --i) {
            int S = gridSizes[i];
            bool isLast = (i == 0);
            cout << "=== Decrypt round " << (rounds - i)
                 << " (grid size: " << S << ") ===\n";
            msg = decrypt(msg, S, isLast);
            cout << "Decrypted segment: " << msg << "\n\n";
        }
        return msg;
    }
};

int main() {
    vector<int> gridSizes;

    while (true) {
        //  -------Level 1: Main Menu --------
        cout<<"\n"
            << "************************\n"
            << "* Menu - Level 1       *\n"
            << "* 1. Encrypt a message *\n"
            << "* 2. Decrypt a message *\n"
            << "* 3. Quit              *\n"
            << "************************\n";

        int level1 = getInt("Enter choice: ");
        if (level1 == 3) {
            cout << "Goodbye!\n";
            break;
        }
        else if (level1 == 1) {
            // -------------- Level 2: Encryption Menu --------------
            string plaintext;
            Encryptor E;
            while (true) {
                cout<<"\n"
                    << "******************************************\n"
                    << "* Menu - Level 2: Encryption             *\n"
                    << "* 1. Enter a message                     *\n"
                    << "* 2. One-round encryption                *\n"
                    << "* 3. Automatic multi-round encryption    *\n"
                    << "* 4. Back                                *\n"
                    << "******************************************\n";

                int encChoice = getInt("Enter choice: ");
                if (encChoice == 4) break;

                if (encChoice == 1) {
                    plaintext = getString("Enter message: ");
                }
                else if (encChoice == 2) {
                    if (plaintext.empty()) {
                        cout << "Please enter a message first.\n";
                        continue;
                    }
                    // ------------ Level 3a: One-round Encryption -------------
                    int gridSize = 0;
                    while (true) {
                        cout<<"\n"
                            << "*********************************************\n"
                            << "* Menu - Level 3: Encryption                *\n"
                            << "* 1. Enter the grid size                    *\n"
                            << "* 2. Automatic grid size                    *\n"
                            << "* 3. Print the grid and the encoded message *\n"
                            << "* 4. Back                                   *\n"
                            << "*********************************************\n";

                        int g3 = getInt("Enter choice: ");
                        if (g3 == 4) break;
                        if (g3 == 1) {
                            gridSize = getInt("Enter grid size (Must be an ODD integer greater than or equal to 3): ");
                        }
                        else if (g3 == 2) {
                            gridSize = calcAutoGrid(plaintext);
                            cout << "Auto-selected grid size: " << gridSize << "\n"
                                 << "Your message has been Encrypted.\n";
                        }
                        else if (g3 == 3) {
                            if (gridSize < 3 || gridSize % 2 == 0) {
                                cout << "Please pick a valid grid size first (Must be an ODD integer greater than or equal to 3).\n";
                            } else {
                                string cipher = E.encryption(plaintext, gridSize, true);
                                gridSizes.clear();               // clear previous history
                                gridSizes.push_back(gridSize);  // save grid size for decryption
                                plaintext = cipher;             // update current encrypted message
                                cout << "* One-round cipher: " << cipher << "\n\n";
                            }
                        }
                        else {
                            cout << "Invalid option.\n";
                        }
                    }
                }
                else if (encChoice == 3) {
                    if (plaintext.empty()) {
                        cout << "Please enter a message first.\n";
                        continue;
                    }
                    // ---------- Level 3b: Multi-round Encryption -----------
                    int rounds = 0;
                    while (true) {
                        cout<<"\n"
                            << "*******************************************\n"
                            << "* Menu - Level 3: Encryption              *\n"
                            << "* 1. Enter the number of rounds           *\n"
                            << "* 2. For each round, print the grid and   *\n"
                               "*      the corresponding encoded message  *\n"
                            << "* 3. Back                                 *\n"
                            << "*******************************************\n";

                        int g3 = getInt("Enter choice: ");
                        if (g3 == 3) break;
                        if (g3 == 1) {
                            rounds = getInt("Enter number of rounds: ");
                        }
                        else if (g3 == 2) {
                            if (rounds <= 0) {
                                cout << "Please set rounds first (option 1).\n";
                            } else {
                                plaintext = E.multiEncryption(plaintext,
                                                                rounds,
                                                                gridSizes);
                                cout << "* Final cipher: " << plaintext << "\n\n";
                            }
                        }
                        else {
                            cout << "Invalid option.\n";
                        }
                    }
                }
                else {
                    cout << "Invalid option.\n";
                }
            }
        }
        else if (level1 == 2) {
            // ------------- Level 2: Decryption Menu --------------
            string ciphertext;
            Decryptor D;
            while (true) {
                cout<<"\n"
                    << "*****************************************\n"
                    << "* Menu - Level 2: Decryption            *\n"
                    << "* 1. Enter a message                    *\n"
                    << "* 2. Enter the round number             *\n"
                    << "* 3. For each round, print the grid and *\n"
                       "*    the corresponding decoded message  *\n"
                    << "* 4. Back                               *\n"
                    << "*****************************************\n";

                int decChoice = getInt("Enter choice: ");
                if (decChoice == 4) break;

                static int rounds = 0;
                if (decChoice == 1) {
                    ciphertext = getString("Enter ciphertext: ");
                }
                else if (decChoice == 2) {
                    rounds = getInt("Enter number of rounds: ");
                }
                else if (decChoice == 3) {
                    if (ciphertext.empty()) {
                        cout << "Please enter ciphertext first.\n";
                    }
                    else if (rounds <= 0) {
                        cout << "Please enter the round number first.\n";
                    }
                    else if ((int)gridSizes.size() != rounds) {
                        cout << "Error: No matching grid history for " << rounds << " round(s).\n"
                            << "Please run encryption again before decrypting.\n";
                    }
                    else {
                        string plain = D.multiRoundDecrypt(ciphertext,
                                                        rounds,
                                                        gridSizes);
                        cout << "* Final plaintext: " << plain << "\n\n";
                    }
                }
                else {
                    cout << "Invalid option.\n";
                }
            }
        }
        else {
            cout << "Invalid option. Please choose 1, 2, or 3.\n";
        }
    }

    return 0;
}