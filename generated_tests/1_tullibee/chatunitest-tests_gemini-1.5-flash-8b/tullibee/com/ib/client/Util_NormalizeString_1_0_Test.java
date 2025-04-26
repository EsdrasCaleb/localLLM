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
    void normalizeString_nullInput_returnsEmptyString() {
        String result = Util.NormalizeString(null);
        assertEquals("", result);
    }

    @Test
    void normalizeString_nonNullInput_returnsInputString() {
        String input = "hello";
        String result = Util.NormalizeString(input);
        assertEquals(input, result);
    }
}
