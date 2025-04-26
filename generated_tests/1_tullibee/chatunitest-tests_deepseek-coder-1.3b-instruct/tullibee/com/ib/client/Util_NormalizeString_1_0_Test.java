package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_NormalizeString_1_0_Test {

    @Test
    void testNormalizeString() {
        // Testing with null
        assertNull(Util.NormalizeString(null));
        // Testing with non-null string
        assertEquals("TestString", Util.NormalizeString("TestString"));
        // Testing with empty string
        assertEquals("", Util.NormalizeString(""));
    }
}
