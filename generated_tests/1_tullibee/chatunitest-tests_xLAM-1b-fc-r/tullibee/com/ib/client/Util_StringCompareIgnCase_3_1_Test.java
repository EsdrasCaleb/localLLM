package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompareIgnCase_3_1_Test {

    @Test
    public void testStringCompareIgnCase() {
        assertEquals(0, Util.StringCompareIgnCase("test", "TEST"));
        assertEquals(-1, Util.StringCompareIgnCase("test", "TEST1"));
        assertEquals(1, Util.StringCompareIgnCase("test1", "test"));
    }
}
