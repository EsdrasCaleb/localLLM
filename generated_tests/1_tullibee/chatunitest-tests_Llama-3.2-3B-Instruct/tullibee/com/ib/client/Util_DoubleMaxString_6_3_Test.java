package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_DoubleMaxString_6_3_Test {

    @Test
    public void testDoubleMaxString_MaxValue() {
        double maxValue = Double.MAX_VALUE;
        String result = Util.DoubleMaxString(maxValue);
        assertEquals("", result);
    }

    @Test
    public void testDoubleMaxString_NotMaxValue() {
        double notMaxValue = 10.0;
        String result = Util.DoubleMaxString(notMaxValue);
        assertEquals("10.0", result);
    }

    @Test
    public void testDoubleMaxString_NullValue() {
        double nullValue = 0.0;
        String result = Util.DoubleMaxString(nullValue);
        assertEquals("0.0", result);
    }

    @Test
    public void testDoubleMaxString_Overflow() {
        double overflowValue = Double.MAX_VALUE + 1;
        String result = Util.DoubleMaxString(overflowValue);
        assertEquals("1.7976931348623157E308", result);
    }

    @Test
    public void testDoubleMaxString_OverflowedValue() {
        double overflowedValue = Double.MAX_VALUE + 1;
        String result = Util.DoubleMaxString(overflowedValue);
        assertEquals("1.7976931348623157E308", result);
    }
}
