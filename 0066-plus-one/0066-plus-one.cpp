class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        bool next_nine = false;
        if (digits.back() == 9){
            next_nine = true;
            digits.back() = 0;
            for (auto i = digits.rbegin() + 1; i != digits.rend(); ++i){
                if (next_nine){
                    if (*i == 9){
                        *i = 0;
                        next_nine = true;
                    }
                    else {
                        *i += 1;
                        next_nine = false;
                    }
                }
            }
            if (next_nine){
                digits[0] = 1;
                digits.push_back(0);
            }
        }
        else {
            digits.back() += 1;
        }
        return digits;
    }
};