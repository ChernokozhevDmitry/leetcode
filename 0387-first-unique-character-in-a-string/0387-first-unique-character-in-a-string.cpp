class Solution {
public:
    int firstUniqChar(std::string s) {
        std::unordered_map<char, int> temp_mapa;
        for(const auto& ch : s){
            ++temp_mapa[ch];
        }
        for(int i = 0; i != s.size(); ++i){
            if (temp_mapa[s[i]] == 1) return i;
        }
        return -1;
    }
};
