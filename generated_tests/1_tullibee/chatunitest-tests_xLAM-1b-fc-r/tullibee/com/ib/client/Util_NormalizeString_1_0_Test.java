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
    public void testNormalizeString() {
        // Testing with a non-null string
        String normalizedString = Util.NormalizeString("Test String");
        assertEquals("Test String", normalizedString);
        // Testing with null string
        normalizedString = Util.NormalizeString(null);
        assertEquals("", normalizedString);
        // Testing with empty string
        normalizedString = Util.NormalizeString("");
        assertEquals("", normalizedString);
    }
}
