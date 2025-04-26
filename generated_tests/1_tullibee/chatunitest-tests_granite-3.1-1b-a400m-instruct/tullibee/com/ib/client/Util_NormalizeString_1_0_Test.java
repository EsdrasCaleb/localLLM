package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_NormalizeString_1_0_Test {

    @Test
    void testNormalizeString() {
        // Test with a non-null string
        assertEquals("", Util.NormalizeString(""));
        // Test with a null string
        assertEquals("", Util.NormalizeString(null));
        // Test with a string that already has no whitespaces
        assertEquals(" ", Util.NormalizeString(" "));
        // Test with a string that has leading and trailing whitespaces
        assertEquals("Hello world", Util.NormalizeString("  Hello world  "));
        // Test with a string that has no whitespaces
        assertEquals("  ", Util.NormalizeString("  "));
    }
}
