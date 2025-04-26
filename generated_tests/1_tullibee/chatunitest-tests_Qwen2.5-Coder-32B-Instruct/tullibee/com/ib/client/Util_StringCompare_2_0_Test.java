package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_StringCompare_2_0_Test {

    @Test
    public void testStringCompare_lhsLessThanRhs() throws Exception {
        int result = invokeStringCompare("apple", "banana");
        assertEquals(-1, result);
    }

    @Test
    public void testStringCompare_lhsGreaterThanRhs() throws Exception {
        int result = invokeStringCompare("banana", "apple");
        assertEquals(1, result);
    }

    @Test
    public void testStringCompare_lhsEqualsRhs() throws Exception {
        int result = invokeStringCompare("apple", "apple");
        assertEquals(0, result);
    }

    @Test
    public void testStringCompare_lhsNull() throws Exception {
        int result = invokeStringCompare(null, "banana");
        assertEquals(-1, result);
    }

    @Test
    public void testStringCompare_rhsNull() throws Exception {
        int result = invokeStringCompare("apple", null);
        assertEquals(1, result);
    }

    @Test
    public void testStringCompare_bothNull() throws Exception {
        int result = invokeStringCompare(null, null);
        assertEquals(0, result);
    }

    private int invokeStringCompare(String lhs, String rhs) throws Exception {
        Method method = Util.class.getDeclaredMethod("StringCompare", String.class, String.class);
        method.setAccessible(true);
        return (int) method.invoke(null, lhs, rhs);
    }
}
