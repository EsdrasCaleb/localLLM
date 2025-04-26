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
        // Test when input is null
        assertEquals("", Util.NormalizeString(null));
        // Test when input is not null
        assertEquals("test", Util.NormalizeString("test"));
    }
}
