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
    public void testDoubleMaxString() {
        // Test with Double.MAX_VALUE
        String resultMax = Util.DoubleMaxString(Double.MAX_VALUE);
        assertEquals("", resultMax, "Expected empty string for Double.MAX_VALUE");
        // Test with a normal value
        String resultNormal = Util.DoubleMaxString(123.45);
        assertEquals("123.45", resultNormal, "Expected string representation of the value");
        // Test with negative normal value
        String resultNegative = Util.DoubleMaxString(-123.45);
        assertEquals("-123.45", resultNegative, "Expected string representation of the negative value");
        // Test with zero
        String resultZero = Util.DoubleMaxString(0.0);
        assertEquals("0.0", resultZero, "Expected string representation of zero");
        // Test with Double.MIN_VALUE
        String resultMin = Util.DoubleMaxString(Double.MIN_VALUE);
        assertEquals("4.9E-324", resultMin, "Expected string representation of Double.MIN_VALUE");
    }
}
