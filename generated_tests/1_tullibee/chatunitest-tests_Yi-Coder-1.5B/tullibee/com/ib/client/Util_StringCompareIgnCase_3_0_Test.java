package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompareIgnCase_3_0_Test {

    @Test
    void test1() {
        assertEquals(0, Util.StringCompareIgnCase("abc", "abc"));
    }

    @Test
    void test2() {
        assertEquals(1, Util.StringCompareIgnCase("abc", "AbC"));
    }

    @Test
    void test3() {
        assertEquals(-1, Util.StringCompareIgnCase("abc", "aBc"));
    }
}
