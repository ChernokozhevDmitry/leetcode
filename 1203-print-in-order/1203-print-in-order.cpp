class Foo {
private:
    mutex mtx;
    condition_variable cv;
    int next = 1;
public:
    void first(function<void()> printFirst) {
        unique_lock unl(mtx);
        // printFirst() outputs "first". Do not change or remove this line.
        printFirst();
        next = 2;
        unl.unlock();
        cv.notify_all();
    }

    void second(function<void()> printSecond) {
        unique_lock unl(mtx);
        cv.wait(unl, [&](){return next == 2;});
        // printSecond() outputs "second". Do not change or remove this line.
        printSecond();
        next = 3;
        unl.unlock();
        cv.notify_all();
    }

    void third(function<void()> printThird) {
        unique_lock unl(mtx);
        cv.wait(unl, [&](){return next == 3;});
        // printThird() outputs "third". Do not change or remove this line.
        printThird();
        unl.unlock();
    }
};