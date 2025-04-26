package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompare_2_0_Test {

    @InjectMocks
    private Util util;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testStringCompareEqual() {
        String lhs = "abc";
        String rhs = "abc";
        int result = Util.StringCompare(lhs, rhs);
        assertEquals(0, result);
    }

    @Test
    public void testStringCompareLessThan() {
        String lhs = "abc";
        String rhs = "def";
        int result = Util.StringCompare(lhs, rhs);
        assertTrue(result < 0);
    }

    @Test
    public void testStringCompareGreaterThan() {
        String lhs = "def";
        String rhs = "abc";
        int result = Util.StringCompare(lhs, rhs);
        assertTrue(result > 0);
    }

    @Test
    public void testStringCompareNull() {
        String lhs = null;
        String rhs = "abc";
        int result = Util.StringCompare(lhs, rhs);
        assertTrue(result < 0);
    }

    @Test
    public void testStringCompareBothNull() {
        String lhs = null;
        String rhs = null;
        int result = Util.StringCompare(lhs, rhs);
        assertEquals(0, result);
    }
}
