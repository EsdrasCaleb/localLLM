package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_IntMaxString_5_0_Test {

    @Test
    void testIntMaxString_maxValue() {
        assertEquals("", Util.IntMaxString(Integer.MAX_VALUE));
    }

    @Test
    void testIntMaxString_otherValue() {
        assertEquals("10", Util.IntMaxString(10));
    }

    @Test
    void testIntMaxString_minValue() {
        assertEquals("-2147483648", Util.IntMaxString(Integer.MIN_VALUE));
    }

    @Test
    void testIntMaxString_zero() {
        assertEquals("0", Util.IntMaxString(0));
    }
}
