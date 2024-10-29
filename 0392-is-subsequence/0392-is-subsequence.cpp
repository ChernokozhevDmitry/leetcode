class Solution {
public:
    bool isSubsequence(string s, string t) {
        int it_source = 0, it_target = 0;
        while ((it_source < s.length())&&(it_target < t.length())){
            if (s[it_source] == t[it_target]){
                ++it_source;
            }
            ++it_target;
        }
        return it_source == s.length();
    }
};