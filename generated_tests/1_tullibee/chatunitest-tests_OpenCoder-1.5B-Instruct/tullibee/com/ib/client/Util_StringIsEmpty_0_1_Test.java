package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_StringIsEmpty_0_1_Test {

    @InjectMocks
    private Util util;

    @Test
    public void testStringIsEmpty() {
        // Test for null string
        assertTrue(util.StringIsEmpty(null));
        // Test for empty string
        assertTrue(util.StringIsEmpty(""));
        // Test for non-empty string
        assertFalse(util.StringIsEmpty("Hello, World!"));
    }
}
