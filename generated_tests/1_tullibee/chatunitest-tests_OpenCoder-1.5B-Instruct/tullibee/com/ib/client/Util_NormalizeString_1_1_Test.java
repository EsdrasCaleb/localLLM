package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_NormalizeString_1_1_Test {

    @Test
    public void testNormalizeString() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        Util util = new Util();
        Method method = Util.class.getDeclaredMethod("NormalizeString", String.class);
        method.setAccessible(true);
        // Test with null string
        String nullString = null;
        String normalizedNull = (String) method.invoke(util, nullString);
        assertEquals("", normalizedNull);
        // Test with non-null string
        String nonNullString = "Hello, World!";
        String normalizedNonNull = (String) method.invoke(util, nonNullString);
        assertEquals("Hello, World!", normalizedNonNull);
    }
}
