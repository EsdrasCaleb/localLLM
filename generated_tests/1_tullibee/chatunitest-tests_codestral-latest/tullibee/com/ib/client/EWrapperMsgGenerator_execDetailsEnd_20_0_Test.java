package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_execDetailsEnd_20_0_Test {

    @Test
    public void testExecDetailsEnd() throws Exception {
        // Create an instance of the class
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        // Use reflection to invoke the private method
        Method method = EWrapperMsgGenerator.class.getDeclaredMethod("execDetailsEnd", int.class);
        method.setAccessible(true);
        // Test with reqId = 1
        String result1 = (String) method.invoke(generator, 1);
        assertEquals("reqId = 1 =============== end ===============", result1);
        // Test with reqId = 100
        String result2 = (String) method.invoke(generator, 100);
        assertEquals("reqId = 100 =============== end ===============", result2);
        // Test with reqId = -1
        String result3 = (String) method.invoke(generator, -1);
        assertEquals("reqId = -1 =============== end ===============", result3);
    }
}
