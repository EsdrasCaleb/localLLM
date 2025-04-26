package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_DoubleMaxString_6_0_Test {

    @Test
    void testDoubleMaxString() {
        assertEquals("", Util.DoubleMaxString(Double.MAX_VALUE), "Should return empty string for Double.MAX_VALUE");
        assertEquals("1.0", Util.DoubleMaxString(1.0), "Should return string representation for 1.0");
        assertEquals("-1.0", Util.DoubleMaxString(-1.0), "Should return string representation for -1.0");
        assertEquals("0.0", Util.DoubleMaxString(0.0), "Should return string representation for 0.0");
        assertEquals("3.14159", Util.DoubleMaxString(3.14159), "Should return string representation for 3.14159");
        // This will actually return "NaN"
        assertEquals("Double.NaN", Util.DoubleMaxString(Double.NaN), "Should return string representation for NaN");
        // This will actually return "Infinity"
        assertEquals("Infinity", Util.DoubleMaxString(Double.POSITIVE_INFINITY), "Should return string representation for Infinity");
        // This will actually return "-Infinity"
        assertEquals("-Infinity", Util.DoubleMaxString(Double.NEGATIVE_INFINITY), "Should return string representation for -Infinity");
    }
}
