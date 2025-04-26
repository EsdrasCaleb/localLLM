package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompare_2_0_Test {

    @Test
    public void testStringCompare_LessThan() throws Exception {
        String lhs = "apple";
        String rhs = "banana";
        int result = invokeStringCompare(lhs, rhs);
        assertTrue(result < 0);
    }

    @Test
    public void testStringCompare_Equal() throws Exception {
        String lhs = "cherry";
        String rhs = "cherry";
        int result = invokeStringCompare(lhs, rhs);
        assertEquals(0, result);
    }

    @Test
    public void testStringCompare_GreaterThan() throws Exception {
        String lhs = "date";
        String rhs = "apple";
        int result = invokeStringCompare(lhs, rhs);
        assertTrue(result > 0);
    }

    @Test
    public void testStringCompare_EmptyStrings() throws Exception {
        String lhs = "";
        String rhs = "";
        int result = invokeStringCompare(lhs, rhs);
        assertEquals(0, result);
    }

    @Test
    public void testStringCompare_NullAndEmpty() throws Exception {
        String lhs = null;
        String rhs = "";
        int result = invokeStringCompare(lhs, rhs);
        assertTrue(result > 0);
    }

    private int invokeStringCompare(String lhs, String rhs) throws Exception {
        Method method = Util.class.getDeclaredMethod("StringCompare", String.class, String.class);
        method.setAccessible(true);
        return (int) method.invoke(null, lhs, rhs);
    }
}
