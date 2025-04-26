package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_DoubleMaxString_6_1_Test {

    // Test cases
    @Test
    public void testDoubleMaxString1() {
        assertEquals("", Util.DoubleMaxString(0));
    }

    @Test
    public void testDoubleMaxString2() {
        assertEquals("1", Util.DoubleMaxString(1));
    }

    @Test
    public void testDoubleMaxString3() {
        assertEquals("1000000", Util.DoubleMaxString(1000000));
    }

    @Test
    public void testDoubleMaxString4() {
        assertEquals("", Util.DoubleMaxString(Double.MAX_VALUE));
    }
}
