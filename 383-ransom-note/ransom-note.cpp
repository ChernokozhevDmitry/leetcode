class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        bool yesorno = false;
        for (char sub : ransomNote) {
            yesorno = false; 
            for (size_t i = 0; i != magazine.size(); ++i) {
                if (magazine[i] == sub) {
                    magazine.erase(i,1);
                    yesorno = true; 
                    break;
                }
            }
            if (!yesorno) {
                break;
            }
        }
        return yesorno;    
    }
};