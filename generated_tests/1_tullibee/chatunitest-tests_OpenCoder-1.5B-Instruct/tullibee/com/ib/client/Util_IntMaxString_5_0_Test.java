package com.ib.client;

import java.lang.reflect.Method;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Util_IntMaxString_5_0_Test {

    private static final Method intMaxStringMethod;

    static {
        try {
            intMaxStringMethod = Util.class.getDeclaredMethod("IntMaxString", int.class);
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
        intMaxStringMethod.setAccessible(true);
    }

    @Test
    void testIntMaxString() throws Exception {
        Object result = intMaxStringMethod.invoke(null, Integer.MAX_VALUE);
        assertEquals("", result);
    }
}
