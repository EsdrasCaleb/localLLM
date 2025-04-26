package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompare_2_2_Test {

    @Test
    public void testStringCompare() {
        Util util = new Util();
        String lhs = "abc";
        String rhs = "abc";
        int expected = 0;
        int result = util.StringCompare(lhs, rhs);
        assertEquals(expected, result);
    }
}
