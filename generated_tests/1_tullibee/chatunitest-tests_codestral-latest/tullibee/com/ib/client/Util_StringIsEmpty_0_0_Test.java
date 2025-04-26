package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringIsEmpty_0_0_Test {

    @Test
    public void testStringIsEmpty() {
        // Test case 1: String is null
        assertTrue(Util.StringIsEmpty(null));
        // Test case 2: String is empty
        assertTrue(Util.StringIsEmpty(""));
        // Test case 3: String is not empty
        assertFalse(Util.StringIsEmpty("Hello"));
        // Test case 4: String contains only whitespace
        assertFalse(Util.StringIsEmpty(" "));
    }
}
