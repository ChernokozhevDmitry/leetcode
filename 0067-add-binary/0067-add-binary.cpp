class Solution {
public:
    string addBinary(string a, string b) {
        std::string res;
        int one_move = false;
        int i = a.length() - 1;
        int j = b.length() - 1;

        while (i >= 0 || j >= 0 || one_move) {
            if (i >= 0)
                one_move += a[i--] - '0';
            if (j >= 0)
                one_move += b[j--] - '0';
            res += one_move % 2 + '0';
            one_move /= 2;
        }
        reverse(begin(res), end(res));
        return res;
    }
};