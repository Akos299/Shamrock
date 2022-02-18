#pragma once

template <class T> inline bool test_eq2(T a, T b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    return eqx && eqy;
}

template <class T> inline bool test_eq3(T a, T b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    bool eqz = a.z() == b.z();
    return eqx && eqy && eqz;
}

template <class T> inline bool test_eq4(T a, T b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    bool eqz = a.z() == b.z();
    bool eqw = a.w() == b.w();
    return eqx && eqy && eqz && eqw;
}

template <class T> inline bool test_eq8(T a, T b) {
    bool eqs0 = a.s0() == b.s0();
    bool eqs1 = a.s1() == b.s1();
    bool eqs2 = a.s2() == b.s2();
    bool eqs3 = a.s3() == b.s3();
    bool eqs4 = a.s4() == b.s4();
    bool eqs5 = a.s5() == b.s5();
    bool eqs6 = a.s6() == b.s6();
    bool eqs7 = a.s7() == b.s7();
    return eqs0 && eqs1 && eqs2 && eqs3 && eqs4 && eqs5 && eqs6 && eqs7;
}

template <class T> inline bool test_eq16(T a, T b) {
    bool eqs0 = a.s0() == b.s0();
    bool eqs1 = a.s1() == b.s1();
    bool eqs2 = a.s2() == b.s2();
    bool eqs3 = a.s3() == b.s3();
    bool eqs4 = a.s4() == b.s4();
    bool eqs5 = a.s5() == b.s5();
    bool eqs6 = a.s6() == b.s6();
    bool eqs7 = a.s7() == b.s7();

    bool eqs8 = a.s8() == b.s8();
    bool eqs9 = a.s9() == b.s9();
    bool eqsA = a.sA() == b.sA();
    bool eqsB = a.sB() == b.sB();
    bool eqsC = a.sC() == b.sC();
    bool eqsD = a.sD() == b.sD();
    bool eqsE = a.sE() == b.sE();
    bool eqsF = a.sF() == b.sF();

    return eqs0 && eqs1 && eqs2 && eqs3 && eqs4 && eqs5 && eqs6 && eqs7 && eqs8 && eqs9 && eqsA && eqsB && eqsC && eqsD &&
           eqsE && eqsF;
}