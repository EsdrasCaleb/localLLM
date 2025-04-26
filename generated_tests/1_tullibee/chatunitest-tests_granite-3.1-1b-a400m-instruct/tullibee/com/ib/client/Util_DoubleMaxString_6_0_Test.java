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
        assertEquals("", Util.DoubleMaxString(Double.MIN_VALUE));
        assertEquals("", Util.DoubleMaxString(Double.MAX_VALUE));
        assertEquals("", Util.DoubleMaxString(Double.MIN_VALUE));
        assertEquals("Double.MAX_VALUE", Util.DoubleMaxString(Double.MAX_VALUE));
        assertEquals("Double.MAX_VALUE", Util.DoubleMaxString(Double.MAX_VALUE));
        assertEquals("", Util.DoubleMaxString(Double.MIN_VALUE));
        assertEquals("Double.MAX_VALUE", Util.DoubleMaxString(Double.MIN_VALUE));
    }
}
