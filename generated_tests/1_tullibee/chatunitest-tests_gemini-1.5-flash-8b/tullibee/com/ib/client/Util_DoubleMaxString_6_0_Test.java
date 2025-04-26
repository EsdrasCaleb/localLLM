package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_DoubleMaxString_6_0_Test {

    @Test
    void testDoubleMaxString_maxValue() {
        assertEquals("", Util.DoubleMaxString(Double.MAX_VALUE));
    }

    @Test
    void testDoubleMaxString_normalValue() {
        assertEquals("3.14", Util.DoubleMaxString(3.14));
    }

    @Test
    void testDoubleMaxString_negativeValue() {
        assertEquals("-10.5", Util.DoubleMaxString(-10.5));
    }

    @Test
    void testDoubleMaxString_zero() {
        assertEquals("0.0", Util.DoubleMaxString(0.0));
    }

    @Test
    void testDoubleMaxString_null() {
        // Handles null input gracefully
        assertNull(Util.DoubleMaxString(Double.NaN));
    }
}
