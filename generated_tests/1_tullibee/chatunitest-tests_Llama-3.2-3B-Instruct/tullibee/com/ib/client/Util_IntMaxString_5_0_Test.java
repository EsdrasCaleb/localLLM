package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_IntMaxString_5_0_Test {

    @Test
    public void testIntMaxString_NormalValue() {
        assertEquals("10", Util.IntMaxString(10));
    }

    @Test
    public void testIntMaxString_MaxValue() {
        assertEquals("", Util.IntMaxString(Integer.MAX_VALUE));
    }

    @Test
    public void testIntMaxString_MinValue() {
        assertEquals("Integer.MIN_VALUE", Util.IntMaxString(Integer.MIN_VALUE));
    }

    @Test
    public void testIntMaxString_Zero() {
        assertEquals("0", Util.IntMaxString(0));
    }

    @Test
    public void testIntMaxString_NegativeValue() {
        assertEquals("-10", Util.IntMaxString(-10));
    }
}
