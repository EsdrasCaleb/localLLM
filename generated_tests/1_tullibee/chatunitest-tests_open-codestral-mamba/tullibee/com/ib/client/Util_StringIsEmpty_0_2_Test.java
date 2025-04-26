package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringIsEmpty_0_2_Test {

    @Test
    public void testStringIsEmpty() {
        // Test with null input
        assertTrue(Util.StringIsEmpty(null));
        // Test with empty string
        assertTrue(Util.StringIsEmpty(""));
        // Test with non-empty string
        assertFalse(Util.StringIsEmpty("Hello"));
        // Test with whitespace string
        assertFalse(Util.StringIsEmpty(" "));
    }
}
