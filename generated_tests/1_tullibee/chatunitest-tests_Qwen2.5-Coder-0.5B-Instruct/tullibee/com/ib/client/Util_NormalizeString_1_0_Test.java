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
        // Test cases
        assertDoesNotThrow(() -> Util.NormalizeString(null));
        assertDoesNotThrow(() -> Util.NormalizeString(" Leading whitespace   "));
        assertDoesNotThrow(() -> Util.NormalizeString("Trailing whitespace   "));
        assertDoesNotThrow(() -> Util.NormalizeString("Mixed whitespace   "));
        assertDoesNotThrow(() -> Util.NormalizeString("   Leading and trailing whitespace   "));
        assertDoesNotThrow(() -> Util.NormalizeString("   Leading and trailing whitespace   "));
    }
}
