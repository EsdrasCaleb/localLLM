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
    void testNormalizeStringNull() {
        String result = Util.NormalizeString(null);
        assertEquals("", result);
    }

    @Test
    void testNormalizeStringEmpty() {
        String result = Util.NormalizeString("");
        assertEquals("", result);
    }

    @Test
    void testNormalizeStringNonEmpty() {
        String result = Util.NormalizeString("test");
        assertEquals("test", result);
    }

    @Test
    void testNormalizeStringWhitespace() {
        String result = Util.NormalizeString("   ");
        assertEquals("   ", result);
    }
}
