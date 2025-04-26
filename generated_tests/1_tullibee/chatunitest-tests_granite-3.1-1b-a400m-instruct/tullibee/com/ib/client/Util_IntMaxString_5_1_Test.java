package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_IntMaxString_5_1_Test {

    @Test
    public void testIntMaxString() {
        assertEquals("Integer.MAX_VALUE", Util.IntMaxString(Integer.MAX_VALUE));
        assertEquals("", Util.IntMaxString(Integer.MIN_VALUE));
        assertEquals("Integer.MAX_VALUE", Util.IntMaxString(Integer.MAX_VALUE));
    }
}
