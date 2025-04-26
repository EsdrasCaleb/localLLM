package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompare_2_0_Test {

    @Test
    public void testStringCompare() {
        assertEquals(-1, Util.StringCompare("apple", "banana"));
        assertEquals(1, Util.StringCompare("banana", "apple"));
        assertEquals(0, Util.StringCompare("apple", "apple"));
    }
}
