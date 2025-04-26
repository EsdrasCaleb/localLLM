package com.ib.client;

import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompareIgnCase_3_0_Test {

    @Test
    public void testStringCompareIgnCase() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        // Create an instance of the class being tested
        Util util = new Util();
        // Get the method being tested
        Method method = Util.class.getDeclaredMethod("StringCompareIgnCase", String.class, String.class);
        // Make the method accessible
        method.setAccessible(true);
        // Invoke the method and store the result
        int result = (int) method.invoke(util, "Hello", "hello");
        // Assert that the result is 0
        assertEquals(0, result);
    }
}
