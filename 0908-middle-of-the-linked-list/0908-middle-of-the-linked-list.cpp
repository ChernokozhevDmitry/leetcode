/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        int i = 0;
        ListNode* ptr = head;
        while(ptr) {
            ptr = ptr->next;
            ++i;
        }
        (i/= 2)++;
        int j = 1;
        ptr = head;
        while(j < i) {
            ptr = ptr->next;
            ++j;
        }
        return ptr;
    }
};